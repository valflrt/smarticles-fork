use std::{
    sync::mpsc::Receiver,
    thread::sleep,
    time::{Duration, Instant},
};

use eframe::egui::{Context, Vec2};

use crate::{
    ai::{
        net::Network,
        training::{
            adapt_input, apply_output, calc_geometric_center, get_avg_distance_between_particles,
            setup_simulation_for_networks, INFERENCE_TICK_INTERVAL,
        },
    },
    events::{Event, Recipient, StateUpdate},
    simulation::Simulation,
    Senders,
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
    last_geometric_center: Vec2,

    senders: Senders,
    receiver: Receiver<Event>,
}

impl SimulationManager {
    pub fn start(senders: Senders, receiver: Receiver<Event>, ctx: Context) {
        let mut slf = Self {
            simulation_state: SimulationState::Paused,

            simulation: Simulation::default(),

            network: None,
            steps: 0,
            network_state: NetworkState::Stopped,
            target_angle: 0.,
            last_geometric_center: Vec2::ZERO,

            senders,
            receiver,
        };

        // sleep(Duration::from_millis(500));

        while slf.update() {
            ctx.request_repaint();
        }
    }

    pub fn update(&mut self) -> bool {
        let events = self.receiver.try_iter().collect::<Vec<_>>();
        for event in events {
            match event {
                Event::StateUpdate(StateUpdate {
                    power_matrix,
                    particle_counts,

                    inference_network,
                    target_angle,
                    ..
                }) => {
                    if let Some(power_matrix) = power_matrix {
                        self.simulation.power_matrix = power_matrix;
                    }
                    if let Some(particle_counts) = particle_counts {
                        self.simulation.particle_counts = particle_counts;
                    }

                    self.network = inference_network;
                    if let Some(target_angle) = target_angle {
                        self.target_angle = target_angle;
                    }
                }

                Event::SpawnParticles => {
                    self.simulation.spawn();
                    self.senders.send(
                        Recipient::App,
                        Event::StateUpdate(
                            StateUpdate::new()
                                .particle_positions(&self.simulation.particle_positions)
                                .particle_counts(&self.simulation.particle_counts),
                        ),
                    )
                }

                Event::SimulationStart => self.simulation_state = SimulationState::Running,
                Event::SimulationPause => self.simulation_state = SimulationState::Paused,

                Event::NetworkStart => {
                    setup_simulation_for_networks(&mut self.simulation);
                    // custom_particule_spawn(self.target_angle, &mut self.simulation);
                    self.simulation.spawn();
                    self.senders.send(
                        Recipient::App,
                        Event::StateUpdate(
                            StateUpdate::new()
                                .particle_positions(&self.simulation.particle_positions)
                                .particle_counts(&self.simulation.particle_counts),
                        ),
                    );

                    self.network_state = NetworkState::Running;
                }
                Event::NetworkStop => self.network_state = NetworkState::Stopped,

                Event::Exit => return false,
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
                        let avg_distance_between_particles =
                            get_avg_distance_between_particles(&self.simulation);
                        let gc = calc_geometric_center(&self.simulation);

                        let movement = gc - self.last_geometric_center;

                        let output = network.infer(adapt_input(
                            self.target_angle,
                            movement,
                            movement.x * self.target_angle.cos()
                                + movement.y * self.target_angle.sin(),
                            avg_distance_between_particles,
                        ));
                        apply_output(output, &mut self.simulation.power_matrix);

                        self.senders.send(
                            Recipient::App,
                            Event::StateUpdate(
                                StateUpdate::new().power_matrix(&self.simulation.power_matrix),
                            ),
                        )
                    } else {
                        self.steps += 1;
                    }
                }
            }

            self.senders.send(
                Recipient::App,
                Event::StateUpdate(
                    StateUpdate::new()
                        .particle_positions(&self.simulation.particle_positions)
                        .computation_duration(elapsed),
                ),
            );

            if elapsed < UPDATE_INTERVAL {
                sleep(UPDATE_INTERVAL - elapsed);
            }
        } else {
            sleep(PAUSED_UPDATE_INTERVAL);
        }

        true
    }
}
