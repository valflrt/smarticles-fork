use std::{
    sync::mpsc::Receiver,
    thread::sleep,
    time::{Duration, Instant},
};

use egui::Vec2;
use log::debug;

use crate::{
    ai::net::Network,
    app::training::{
        adapt_input, calc_geometric_centers_and_mean_distances, setup_simulation_for_network,
        INFERENCE_TICK_INTERVAL,
    },
    simulation::Simulation,
    Senders, SmarticlesEvent, MAX_FORCE,
};

/// Min update interval in ms (when the simulation is running).
const UPDATE_INTERVAL: Duration = Duration::from_millis(30);
/// Min update rate when the simulation is paused.
const PAUSED_UPDATE_INTERVAL: Duration = Duration::from_millis(200);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimulationState {
    Paused,
    Running,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NetworkState {
    Stopped,
    Running,
}

pub struct SimulationBackend {
    simulation_state: SimulationState,

    simulation: Simulation,

    network: Option<Network>,
    steps: u8,
    network_state: NetworkState,
    target_position: Vec2,

    senders: Senders,
    receiver: Receiver<SmarticlesEvent>,
}

impl SimulationBackend {
    pub fn new(senders: Senders, receiver: Receiver<SmarticlesEvent>) -> Self {
        Self {
            simulation_state: SimulationState::Paused,

            simulation: Simulation::default(),

            network: None,
            steps: 0,
            network_state: NetworkState::Stopped,
            target_position: Vec2::ZERO,

            senders,
            receiver,
        }
    }

    pub fn update(&mut self) -> bool {
        let events = self.receiver.try_iter().collect::<Vec<_>>();
        debug!("Received events {:?}", events);
        for event in events {
            match event {
                SmarticlesEvent::Quit => return false,

                SmarticlesEvent::SpawnParticles => {
                    self.simulation.spawn();
                    self.senders.send_ui(SmarticlesEvent::SimulationResults(
                        self.simulation.particle_positions.to_owned(),
                        None,
                    ));
                }

                SmarticlesEvent::SimulationStart => {
                    self.simulation_state = SimulationState::Running
                }
                SmarticlesEvent::SimulationPause => self.simulation_state = SimulationState::Paused,

                SmarticlesEvent::InferenceNetworkChange(network) => {
                    self.network = Some(network);
                }
                SmarticlesEvent::NetworkStart => {
                    setup_simulation_for_network(&mut self.simulation);
                    self.senders.send_ui(SmarticlesEvent::ParticleCountsUpdate(
                        self.simulation.particle_counts,
                    ));
                    self.simulation.spawn();
                    self.senders.send_ui(SmarticlesEvent::SimulationResults(
                        self.simulation.particle_positions.to_owned(),
                        None,
                    ));
                    self.network_state = NetworkState::Running;
                }
                SmarticlesEvent::NetworkStop => self.network_state = NetworkState::Stopped,

                SmarticlesEvent::ForceMatrixChange(force_matrix) => {
                    self.simulation.force_matrix = force_matrix
                }
                SmarticlesEvent::ParticleCountsUpdate(particle_counts) => {
                    self.simulation.particle_counts = particle_counts
                }

                SmarticlesEvent::SimulationReset => {
                    self.simulation_state = SimulationState::Paused;
                    self.simulation.reset_particles_positions();
                }

                SmarticlesEvent::TargetPositionChange(target_position) => {
                    self.target_position = target_position
                }

                _ => {}
            }
        }

        if self.simulation_state == SimulationState::Running {
            let start_time = Instant::now();
            self.simulation.move_particles();
            let elapsed = start_time.elapsed();

            if self.network_state == NetworkState::Running {
                if let Some(network) = &self.network {
                    if self.steps == INFERENCE_TICK_INTERVAL {
                        let (gcs, ggc) =
                            calc_geometric_centers_and_mean_distances(&self.simulation);

                        let ggc_to_target_direction = self.target_position - ggc;

                        let mut output = network.infer(adapt_input(
                            ggc_to_target_direction.normalized(),
                            gcs,
                            self.simulation.force_matrix.to_owned(),
                        ));

                        output.iter_mut().for_each(|x| *x *= MAX_FORCE);
                        *self.simulation.force_matrix.vec_mut() = output;

                        self.senders.send_ui(SmarticlesEvent::ForceMatrixChange(
                            self.simulation.force_matrix.to_owned(),
                        ));
                        self.steps = 0;
                    } else {
                        self.steps += 1;
                    }
                }
            }

            self.senders.send_ui(SmarticlesEvent::SimulationResults(
                self.simulation.particle_positions.to_owned(),
                Some(elapsed),
            ));

            if elapsed < UPDATE_INTERVAL {
                sleep(UPDATE_INTERVAL - elapsed);
            }
        } else {
            debug!("simulation paused, update interval reduced");
            sleep(PAUSED_UPDATE_INTERVAL);
        }

        true
    }
}
