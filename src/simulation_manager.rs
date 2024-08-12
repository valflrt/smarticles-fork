use std::{
    sync::mpsc::Receiver,
    thread::sleep,
    time::{Duration, Instant},
};

use crate::{simulation::Simulation, Senders, SmarticlesEvent};

/// Min update interval in ms (when the simulation is running).
const UPDATE_INTERVAL: Duration = Duration::from_millis(30);
/// Min update rate when the simulation is paused.
const PAUSED_UPDATE_INTERVAL: Duration = Duration::from_millis(200);

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SimulationState {
    Paused,
    Running,
}

pub struct SimulationManager {
    simulation_state: SimulationState,

    simulation: Simulation,

    senders: Senders,
    receiver: Receiver<SmarticlesEvent>,
}

impl SimulationManager {
    pub fn new(senders: Senders, receiver: Receiver<SmarticlesEvent>) -> Self {
        Self {
            simulation_state: SimulationState::Paused,

            simulation: Simulation::default(),

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
                    self.senders.send_ui(SmarticlesEvent::SimulationResults(
                        self.simulation.particle_positions.to_owned(),
                        None,
                    ));
                }

                SmarticlesEvent::SimulationStart => {
                    self.simulation_state = SimulationState::Running
                }
                SmarticlesEvent::SimulationPause => self.simulation_state = SimulationState::Paused,

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

                _ => {}
            }
        }

        if self.simulation_state == SimulationState::Running {
            let start_time = Instant::now();
            self.simulation.move_particles();
            let elapsed = start_time.elapsed();

            self.senders.send_ui(SmarticlesEvent::SimulationResults(
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
