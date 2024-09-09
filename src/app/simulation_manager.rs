use std::{
    sync::mpsc::Receiver,
    thread::sleep,
    time::{Duration, Instant},
};

use crate::{
    ai::{
        net::Network,
        training::{
            adapt_input, apply_output, setup_simulation_for_networks, INFERENCE_TICK_INTERVAL,
        },
    },
    simulation::Simulation,
    Senders, SmarticlesEvent, CLASS_COUNT,
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

pub struct SimulationManager {
    simulation_state: SimulationState,

    simulation: Simulation,

    network: Option<Network>,
    steps: usize,
    network_state: NetworkState,
    target_angle: f32,

    senders: Senders,
    receiver: Receiver<SmarticlesEvent>,
}

impl SimulationManager {
    pub fn new(senders: Senders, receiver: Receiver<SmarticlesEvent>) -> Self {
        Self {
            simulation_state: SimulationState::Paused,

            simulation: Simulation::default(),

            network: None,
            steps: 0,
            network_state: NetworkState::Stopped,
            target_angle: 0.,

            senders,
            receiver,
        }
    }

    pub fn update(&mut self) -> bool {
        let events = self.receiver.try_iter().collect::<Vec<_>>();
        for event in events {
            match event {
                SmarticlesEvent::Quit => return false,

                SmarticlesEvent::SpawnParticles => {
                    self.simulation.spawn();
                    self.senders.send_to_app(SmarticlesEvent::SimulationResults(
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
                    setup_simulation_for_networks(&mut self.simulation);
                    self.senders
                        .send_to_app(SmarticlesEvent::ParticleCountsUpdate(
                            self.simulation.particle_counts,
                        ));
                    self.senders.send_to_app(SmarticlesEvent::SimulationResults(
                        self.simulation.particle_positions.to_owned(),
                        None,
                    ));
                    self.network_state = NetworkState::Running;
                }
                SmarticlesEvent::NetworkStop => self.network_state = NetworkState::Stopped,

                SmarticlesEvent::PowerMatrixChange(power_matrix) => {
                    self.simulation.power_matrix = power_matrix
                }
                SmarticlesEvent::ParticleCountsUpdate(particle_counts) => {
                    self.simulation.particle_counts = particle_counts
                }

                SmarticlesEvent::SimulationReset => {
                    self.simulation_state = SimulationState::Paused;
                    self.simulation.particle_counts = [0; CLASS_COUNT];
                    self.simulation.reset_particles_positions();
                }

                SmarticlesEvent::TargetAngleChange(target_angle) => {
                    self.target_angle = target_angle
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
                    if self.steps % INFERENCE_TICK_INTERVAL == 0 {
                        let output = network.infer(adapt_input(
                            self.target_angle,
                            self.simulation.power_matrix.to_owned(),
                        ));
                        apply_output(output, &mut self.simulation.power_matrix);

                        self.senders.send_to_app(SmarticlesEvent::PowerMatrixChange(
                            self.simulation.power_matrix.to_owned(),
                        ));
                    } else {
                        self.steps += 1;
                    }
                }
            }

            self.senders.send_to_app(SmarticlesEvent::SimulationResults(
                self.simulation.particle_positions.to_owned(),
                Some(elapsed),
            ));

            if elapsed < UPDATE_INTERVAL {
                sleep(UPDATE_INTERVAL - elapsed);
            }
        } else {
            sleep(PAUSED_UPDATE_INTERVAL);
        }

        true
    }
}
