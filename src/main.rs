mod app;
mod mat;
mod simulation;
mod simulation_manager;

use std::{
    sync::mpsc::{channel, Sender},
    thread,
    time::Duration,
};

use eframe::{
    egui::{Vec2, ViewportBuilder},
    epaint::{Color32, Hsva},
    NativeOptions,
};
use mat::Mat2D;

use crate::{app::SmarticlesApp, simulation_manager::SimulationManager};

#[cfg(feature = "cell_map_display")]
use simulation::Cell;

// IDEA Add recordings ? By exporting positions of all the
// particles each frame ? That would make around 8000 postions
// every 1/60 second that is to say 60*8000=480,000 positions
// per second, let's assume a position is 8 bytes (from Vec2),
// then one second of simulation is 8*480,000=3,840,000 bytes
// this is around 4MB. 1min of simulation is 60*4=240MB.
// This seems possible, although not for long recordings.
// Saving the exact starting position might also work although
// if the simulation runs for too long there might be differences
// between computers.

const CLASS_COUNT: usize = 6;

/// Min particle count.
const MIN_PARTICLE_COUNT: usize = 0;
/// Maximal particle count per class.
const MAX_PARTICLE_COUNT: usize = 15000;
/// When randomizing particle counts, this is the lowest
/// possible value, this prevent random particle counts from
/// being under this value.
const RANDOM_MIN_PARTICLE_COUNT: usize = 200;
/// When randomizing particle counts, this is the highest
/// possible value, this prevent random particle counts from
/// being above this value.
const RANDOM_MAX_PARTICLE_COUNT: usize = MAX_PARTICLE_COUNT;

const MAX_POWER: i8 = 100;
const MIN_POWER: i8 = -MAX_POWER;

fn main() {
    let options = NativeOptions {
        viewport: ViewportBuilder::default().with_fullscreen(true),
        ..Default::default()
    };

    let (ui_sender, ui_receiver) = channel::<SmarticlesEvent>();
    let (sim_sender, sim_receiver) = channel::<SmarticlesEvent>();

    let senders = Senders::new(ui_sender, sim_sender);

    eframe::run_native(
        "Smarticles",
        options,
        Box::new(|cc| {
            let frame = cc.egui_ctx.clone();

            let senders_clone = senders.clone();
            let simulation_handle = thread::spawn(move || {
                let mut sim_backend = SimulationManager::new(senders_clone, sim_receiver);

                thread::sleep(Duration::from_millis(500));

                loop {
                    if !sim_backend.update() {
                        break;
                    };
                    frame.request_repaint();
                }
            });

            Ok(Box::new(SmarticlesApp::new(
                ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ"]
                    .iter()
                    .take(CLASS_COUNT)
                    .copied()
                    .enumerate()
                    .map(|(i, class_name)| {
                        let [r, g, b] = Hsva::new(
                            // tinkering to make purple appear first. I like purple.
                            0.2 + 0.8 * (((CLASS_COUNT - i - 1) % CLASS_COUNT) as f32)
                                / (CLASS_COUNT as f32),
                            0.9,
                            0.9,
                            1.,
                        )
                        .to_srgb();
                        (class_name, Color32::from_rgb(r, g, b))
                    })
                    .collect::<Vec<(&str, Color32)>>()
                    .try_into()
                    .unwrap(),
                senders,
                ui_receiver,
                Some(simulation_handle),
            )))
        }),
    )
    .unwrap();
}

#[derive(Debug, Clone)]
struct Senders {
    ui_sender: Sender<SmarticlesEvent>,
    sim_sender: Sender<SmarticlesEvent>,
}

impl Senders {
    pub fn new(ui_sender: Sender<SmarticlesEvent>, sim_sender: Sender<SmarticlesEvent>) -> Self {
        Senders {
            ui_sender,
            sim_sender,
        }
    }

    pub fn send_ui(&self, event: SmarticlesEvent) {
        self.ui_sender.send(event).unwrap()
    }
    pub fn send_sim(&self, event: SmarticlesEvent) {
        self.sim_sender.send(event).unwrap()
    }
}

#[derive(Debug, Clone)]
enum SmarticlesEvent {
    Quit,

    SpawnParticles,

    /// Particle positions and elapsed time (if available).
    SimulationResults(Mat2D<Vec2>, Option<Duration>),

    #[cfg(feature = "cell_map_display")]
    CellMap(Vec<Cell>),

    EnableClass(usize),
    DisableClass(usize),
    PowerMatrixChange(Mat2D<i8>),
    ParticleCountsUpdate([usize; CLASS_COUNT]),

    SimulationStart,
    SimulationPause,
    SimulationReset,
}
