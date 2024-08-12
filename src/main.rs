use std::path::{Path, PathBuf};
use std::sync::mpsc::{channel, Sender};
use std::time::Duration;
use std::{fs, thread};

use ai::batch_training::Batch;
use ai::net::{ActivationFn, Layer, Network};
use app::simulation::SimulationBackend;
use app::training::{TrainingBackend, BATCH_SIZE, NETWORK_INPUT_SIZE, NETWORK_OUTPUT_SIZE};
use app::ui::Ui;
use eframe::epaint::Color32;
use eframe::NativeOptions;
use egui::color::Hsva;
use egui::Vec2;
use mat::Mat2D;
use postcard::from_bytes;

mod ai;
mod app;
mod mat;
mod simulation;

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
const MAX_PARTICLE_COUNT: usize = 8000;
/// When randomizing particle counts, this is the lowest
/// possible value, this prevent random particle counts from
/// being under this value.
const RANDOM_MIN_PARTICLE_COUNT: usize = 200;
/// When randomizing particle counts, this is the highest
/// possible value, this prevent random particle counts from
/// being above this value.
const RANDOM_MAX_PARTICLE_COUNT: usize = MAX_PARTICLE_COUNT;

const MAX_FORCE: f32 = 100.;
const MIN_FORCE: f32 = -MAX_FORCE;

fn main() {
    let batch = if Path::new("./batches/batch_gen_0").exists() {
        let mut batch_gen = 0;
        let mut batch_path = Path::new("./batches/batch_gen_0").to_path_buf();
        let mut gap = 20;
        while gap > 0 {
            let path: PathBuf = format!("./batches/batch_gen_{}", batch_gen).into();
            if path.exists() {
                gap = 20;
                batch_path = path;
            } else {
                gap -= 1;
            }
            batch_gen += 1;
        }
        from_bytes::<Batch>(&fs::read(batch_path).unwrap()).unwrap()
    } else {
        let batch = Batch::new(Vec::from_iter((0..BATCH_SIZE).map(|_| {
            Network::new([
                Layer::random(NETWORK_INPUT_SIZE, 12, ActivationFn::LeakyRelu),
                Layer::random(12, NETWORK_OUTPUT_SIZE, ActivationFn::Tanh),
            ])
        })));
        batch.save();
        batch
    };

    let options = NativeOptions {
        // initial_window_size: Some(Vec2::new(1600., 900.)),
        fullscreen: true,
        ..Default::default()
    };

    env_logger::init();

    let (ui_sender, ui_receiver) = channel::<SmarticlesEvent>();
    let (sim_sender, sim_receiver) = channel::<SmarticlesEvent>();
    let (train_sender, train_receiver) = channel::<SmarticlesEvent>();

    let senders = Senders::new(ui_sender, sim_sender, train_sender);

    eframe::run_native(
        "Smarticles",
        options,
        Box::new(|cc| {
            let frame = cc.egui_ctx.clone();

            let generation = batch.generation;

            let senders_clone = senders.clone();
            let training_handle = thread::spawn(move || {
                let mut training_backend =
                    TrainingBackend::new(batch, senders_clone, train_receiver);

                loop {
                    if !training_backend.update() {
                        break;
                    };
                }
            });

            let senders_clone = senders.clone();
            let simulation_handle = thread::spawn(move || {
                let mut sim_backend = SimulationBackend::new(senders_clone, sim_receiver);

                thread::sleep(Duration::from_millis(500));

                loop {
                    if !sim_backend.update() {
                        break;
                    };
                    frame.request_repaint();
                }
            });

            Box::new(Ui::new(
                ["α", "β", "γ", "δ", "ε", "ζ", "η", "θ"]
                    .iter()
                    .take(CLASS_COUNT)
                    .copied()
                    .enumerate()
                    .map(|(i, class_name)| {
                        let [r, g, b] =
                            Hsva::new((i as f32) / (CLASS_COUNT as f32), 0.9, 0.9, 1.).to_srgb();
                        (class_name, Color32::from_rgb(r, g, b))
                    })
                    .collect::<Vec<(&str, Color32)>>()
                    .try_into()
                    .unwrap(),
                generation,
                senders,
                ui_receiver,
                Some(simulation_handle),
                Some(training_handle),
            ))
        }),
    );
}

#[derive(Debug, Clone)]
struct Senders {
    ui_sender: Sender<SmarticlesEvent>,
    sim_sender: Sender<SmarticlesEvent>,
    training_sender: Sender<SmarticlesEvent>,
}

impl Senders {
    pub fn new(
        ui_sender: Sender<SmarticlesEvent>,
        sim_sender: Sender<SmarticlesEvent>,
        training_sender: Sender<SmarticlesEvent>,
    ) -> Self {
        Senders {
            ui_sender,
            sim_sender,
            training_sender,
        }
    }

    pub fn send_ui(&self, event: SmarticlesEvent) {
        self.ui_sender.send(event).unwrap()
    }
    pub fn send_sim(&self, event: SmarticlesEvent) {
        self.sim_sender.send(event).unwrap()
    }
    pub fn send_training(&self, event: SmarticlesEvent) {
        self.training_sender.send(event).unwrap()
    }
}

#[derive(Debug, Clone)]
enum SmarticlesEvent {
    Quit,

    SpawnParticles,

    /// Particle positions and elapsed time (if available).
    SimulationResults(Mat2D<Vec2>, Option<Duration>),

    ForceMatrixChange(Mat2D<f32>),
    ParticleCountsUpdate([usize; CLASS_COUNT]),

    StartTraining(usize),
    GenerationChange(usize),
    NetworkRanking(Vec<(f32, Network)>),
    EvaluateNetworks,

    SimulationStart,
    SimulationPause,
    SimulationReset,

    NetworkStart,
    NetworkStop,

    InferenceNetworkChange(Network),
    TargetPositionChange(Vec2),
}
