use std::{
    collections::{hash_map::DefaultHasher, VecDeque},
    hash::{Hash, Hasher},
    sync::mpsc::Receiver,
    thread::JoinHandle,
};

use byteorder::{ReadBytesExt, WriteBytesExt};
use eframe::{
    egui::{
        Align2, CentralPanel, ComboBox, Context, FontId, PointerButton, ScrollArea, Sense,
        SidePanel, Slider, Stroke, Vec2,
    },
    epaint::Color32,
    App, Frame,
};
use rand::{distributions::Open01, rngs::SmallRng, Rng, SeedableRng};
use rayon::prelude::*;

use crate::{
    mat::Mat2D, simulation_manager::SimulationState, Senders, SmarticlesEvent, CLASS_COUNT,
    MAX_FORCE, MAX_PARTICLE_COUNT, MIN_FORCE, MIN_PARTICLE_COUNT, RANDOM_MAX_PARTICLE_COUNT,
    RANDOM_MIN_PARTICLE_COUNT,
};

/// Display diameter of the particles in the simulation (in
/// pixels).
const PARTICLE_DIAMETER: f32 = 0.6;

const DEFAULT_ZOOM: f32 = 2.;
const MIN_ZOOM: f32 = 0.1;
const MAX_ZOOM: f32 = 30.;
const ZOOM_FACTOR: f32 = 1.05;

const MAX_HISTORY_LEN: usize = 10;

pub struct View {
    zoom: f32,
    pos: Vec2,
    dragging: bool,
    drag_start_pos: Vec2,
    drag_start_view_pos: Vec2,
}

impl View {
    const DEFAULT: View = Self {
        zoom: DEFAULT_ZOOM,
        pos: Vec2::ZERO,
        dragging: false,
        drag_start_pos: Vec2::ZERO,
        drag_start_view_pos: Vec2::ZERO,
    };
}

#[derive(Debug)]
struct ClassProps {
    name: String,
    heading: String,
    color: Color32,
}

pub struct SmarticlesApp {
    classes: [ClassProps; CLASS_COUNT],

    seed: String,

    view: View,

    selected_param: (usize, usize),
    selected_particle: (usize, usize),
    follow_selected_particle: bool,

    history: VecDeque<String>,
    selected_history_entry: usize,

    computation_time: u128,

    words: Vec<String>,

    senders: Senders,
    receiver: Receiver<SmarticlesEvent>,

    simulation_handle: Option<JoinHandle<()>>,

    particle_counts: [usize; CLASS_COUNT],
    locked_particle_counts: bool,
    particle_positions: Mat2D<Vec2>,
    force_matrix: Mat2D<i8>,
    simulation_state: SimulationState,
}

impl SmarticlesApp {
    pub fn new<S>(
        classes: [(S, Color32); CLASS_COUNT],
        senders: Senders,
        receiver: Receiver<SmarticlesEvent>,
        simulation_handle: Option<JoinHandle<()>>,
    ) -> Self
    where
        S: ToString,
    {
        let words = include_str!("words.txt");
        let words: Vec<String> = words
            .par_lines()
            .filter_map(|w| {
                if w.len() > 8 {
                    return None;
                }
                for chr in w.chars() {
                    if !chr.is_ascii_alphabetic() || chr.is_ascii_uppercase() {
                        return None;
                    }
                }
                Some(w.to_string())
            })
            .collect();

        Self {
            seed: "".to_string(),

            classes: classes.map(|(name, color)| ClassProps {
                name: name.to_string(),
                heading: "class ".to_string() + &name.to_string(),
                color,
            }),

            view: View::DEFAULT,

            selected_param: (0, 0),
            selected_particle: (0, 0),
            follow_selected_particle: false,

            history: VecDeque::new(),
            selected_history_entry: 0,

            computation_time: 0,

            words,

            senders,
            receiver,

            simulation_handle,

            particle_counts: [200; CLASS_COUNT],
            locked_particle_counts: false,
            particle_positions: Mat2D::filled_with(Vec2::ZERO, CLASS_COUNT, MAX_PARTICLE_COUNT),
            force_matrix: Mat2D::filled_with(0, CLASS_COUNT, CLASS_COUNT),
            simulation_state: SimulationState::Paused,
        }
    }

    fn apply_seed(&mut self) {
        let mut rand = if self.seed.is_empty() {
            SmallRng::from_entropy()
        } else {
            if self.seed.starts_with('@') {
                if let Ok(bytes) = base64::decode(&self.seed[1..]) {
                    self.apply_custom_seed(&bytes);
                    return;
                }
            }
            let mut hasher = DefaultHasher::new();
            self.seed.hash(&mut hasher);
            SmallRng::seed_from_u64(hasher.finish())
        };
        let mut rand = |min: f32, max: f32| min + (max - min) * rand.sample::<f32, _>(Open01);

        const POW_F: f32 = 1.25;

        for i in 0..CLASS_COUNT {
            if !self.locked_particle_counts {
                self.particle_counts[i] = rand(
                    RANDOM_MIN_PARTICLE_COUNT as f32,
                    RANDOM_MAX_PARTICLE_COUNT as f32,
                ) as usize;
            }
            for j in 0..CLASS_COUNT {
                let pow = rand(MIN_FORCE as f32, MAX_FORCE as f32);
                self.force_matrix[(i, j)] = (pow.signum() * pow.abs().powf(1. / POW_F)) as i8;
            }
        }

        self.send_params();
        self.send_particle_counts();
    }
    fn export_custom_seed(&self) -> String {
        let mut bytes: Vec<u8> = Vec::new();
        self.force_matrix
            .vec()
            .iter()
            .copied()
            .for_each(|force_factor| bytes.write_i8(force_factor).unwrap());

        format!("@{}", base64::encode(bytes))
    }
    fn apply_custom_seed(&mut self, mut bytes: &[u8]) {
        for i in 0..CLASS_COUNT {
            for j in 0..CLASS_COUNT {
                self.force_matrix[(i, j)] = bytes.read_i8().unwrap_or(0);
            }
        }
    }

    fn send_params(&self) {
        self.senders.send_sim(SmarticlesEvent::ForceMatrixChange(
            self.force_matrix.to_owned(),
        ));
    }
    fn send_particle_counts(&self) {
        self.senders.send_sim(SmarticlesEvent::ParticleCountsUpdate(
            self.particle_counts.to_owned(),
        ));
    }

    fn update_history(&mut self) {
        if self
            .history
            .front()
            .and_then(|front| Some(&self.seed != front))
            .unwrap_or(true)
        {
            self.history.push_front(self.seed.to_owned());
            if self.history.len() > MAX_HISTORY_LEN {
                self.history.pop_back();
            }
        }
        self.selected_history_entry = 0;
    }

    fn play(&mut self) {
        self.simulation_state = SimulationState::Running;
        self.senders.send_sim(SmarticlesEvent::SimulationStart);
    }
    fn pause(&mut self) {
        self.simulation_state = SimulationState::Paused;
        self.senders.send_sim(SmarticlesEvent::SimulationPause);
    }
    fn reset(&mut self) {
        self.simulation_state = SimulationState::Paused;
        self.senders.send_sim(SmarticlesEvent::SimulationReset);
    }
    fn spawn(&mut self) {
        self.senders.send_sim(SmarticlesEvent::SpawnParticles);
    }
}

impl App for SmarticlesApp {
    fn update(&mut self, ctx: &Context, _: &mut Frame) {
        let events = self.receiver.try_iter();
        for event in events {
            match event {
                SmarticlesEvent::SimulationResults(positions, elapsed) => {
                    if let Some(elapsed) = elapsed {
                        self.computation_time = elapsed.as_millis();
                    }
                    self.particle_positions = positions;
                }

                SmarticlesEvent::ForceMatrixChange(force_matrix) => {
                    self.force_matrix = force_matrix;
                }

                _ => {}
            }
        }

        SidePanel::left("settings").show(ctx, |ui| {
            ui.heading("settings");
            ui.separator();
            ui.horizontal(|ui| {
                if ui
                    .button("respawn")
                    .on_hover_text("spawn particles again")
                    .clicked()
                {
                    self.spawn();
                }

                if self.simulation_state == SimulationState::Running {
                    if ui
                        .button("pause")
                        .on_hover_text("pause the simulation")
                        .clicked()
                    {
                        self.pause();
                    }
                } else if ui
                    .button("play")
                    .on_hover_text("start the simulation")
                    .clicked()
                {
                    self.play();
                }

                if ui
                    .button("randomize")
                    .on_hover_text("randomly pick a new seed")
                    .clicked()
                {
                    let w1 = rand::random::<usize>() % self.words.len();
                    let w2 = rand::random::<usize>() % self.words.len();
                    let w3 = rand::random::<usize>() % self.words.len();
                    self.seed = format!("{}_{}_{}", self.words[w1], self.words[w2], self.words[w3]);

                    self.update_history();

                    self.apply_seed();
                    self.spawn();
                }

                if ui
                    .button("reset View")
                    .on_hover_text("reset zoom and position")
                    .clicked()
                {
                    self.view = View::DEFAULT;
                }

                if ui
                    .button("reset")
                    .on_hover_text("reset everything")
                    .clicked()
                {
                    self.reset();
                }

                if ui.button("quit").on_hover_text("exit smarticles").clicked() {
                    self.senders.send_sim(SmarticlesEvent::Quit);
                    if let Some(handle) = self.simulation_handle.take() {
                        handle.join().unwrap();
                    };
                    ui.ctx()
                        .send_viewport_cmd(eframe::egui::ViewportCommand::Close);
                }
            });
            ui.horizontal(|ui| {
                ui.label("seed:");
                ui.text_edit_singleline(&mut self.seed);
                if ui.button("apply").clicked() {
                    self.update_history();

                    self.apply_seed();
                    self.spawn();
                }
            });

            ui.horizontal(|ui| {
                ui.label("total particle count:");

                let total_particle_count: usize = self.particle_counts.iter().sum();
                ui.code(total_particle_count.to_string());

                ui.checkbox(&mut self.locked_particle_counts, "lock particle counts");
            });

            ui.horizontal(|ui| {
                ui.label("computation time:");
                ui.code(self.computation_time.to_string() + "ms");
            });

            if self.history.len() > 1 {
                ui.collapsing("seed history", |ui| {
                    if ComboBox::from_id_source("seed history")
                        .width(200.)
                        .show_index(
                            ui,
                            &mut self.selected_history_entry,
                            self.history.len(),
                            |i| self.history[i].to_owned(),
                        )
                        .changed()
                    {
                        self.history[self.selected_history_entry].clone_into(&mut self.seed);
                        self.apply_seed();
                        self.spawn();
                    };
                });
            }

            ui.collapsing("particle inspector", |ui| {
                ui.horizontal(|ui| {
                    ui.label("class:");
                    ComboBox::from_id_source("class").show_index(
                        ui,
                        &mut self.selected_particle.0,
                        self.classes.len(),
                        |i| self.classes[i].heading.to_owned(),
                    );
                    ui.label("particle index:");
                    ui.add(Slider::new(
                        &mut self.selected_particle.1,
                        0..=(self.particle_counts[self.selected_particle.0] - 1),
                    ));
                });

                ui.horizontal(|ui| {
                    ui.label("position:");
                    ui.code(format!(
                        "{:?}",
                        self.particle_positions[self.selected_particle]
                    ));
                });

                ui.horizontal(|ui| {
                    if self.follow_selected_particle {
                        if ui.button("stop following selected particle").clicked() {
                            self.view.pos -= self.particle_positions[self.selected_particle];
                            self.follow_selected_particle = false;
                        }
                    } else if ui.button("focus and follow selected particle").clicked() {
                        self.view.pos *= 0.;
                        self.follow_selected_particle = true;
                    }
                });
            });

            ScrollArea::vertical().show(ui, |ui| {
                for i in 0..CLASS_COUNT {
                    ui.add_space(10.);
                    ui.colored_label(self.classes[i].color, &self.classes[i].heading);
                    ui.separator();

                    ui.horizontal(|ui| {
                        ui.label("particle count:");
                        if ui
                            .add(Slider::new(
                                &mut self.particle_counts[i],
                                MIN_PARTICLE_COUNT..=MAX_PARTICLE_COUNT,
                            ))
                            .changed()
                        {
                            self.spawn();
                            self.send_particle_counts();
                        }
                    });

                    ui.collapsing(self.classes[i].heading.to_owned() + " params", |ui| {
                        ui.horizontal(|ui| {
                            ui.vertical(|ui| {
                                for j in 0..CLASS_COUNT {
                                    ui.horizontal(|ui| {
                                        ui.label("force (");
                                        ui.colored_label(
                                            self.classes[j].color,
                                            &self.classes[j].name,
                                        );
                                        ui.label(")");
                                        if ui
                                            .add(Slider::new(
                                                &mut self.force_matrix[(i, j)],
                                                MIN_FORCE..=MAX_FORCE,
                                            ))
                                            .changed()
                                        {
                                            self.selected_param = (i, j);
                                            self.seed = self.export_custom_seed();

                                            self.send_params();
                                        }
                                    });
                                }
                            });
                        });
                    });
                }
            });
        });

        CentralPanel::default()
            .frame(eframe::egui::Frame {
                fill: Color32::from_rgba_unmultiplied(10, 10, 10, 255),
                ..Default::default()
            })
            .show(ctx, |ui| {
                let (resp, paint) =
                    ui.allocate_painter(ui.available_size_before_wrap(), Sense::hover());

                if resp
                    .rect
                    .contains(ctx.input(|i| i.pointer.interact_pos()).unwrap_or_default())
                {
                    let scroll_delta = ctx.input(|i| i.smooth_scroll_delta).y;
                    if scroll_delta > 0. {
                        self.view.zoom *= ZOOM_FACTOR;
                    } else if scroll_delta < 0. {
                        self.view.zoom /= ZOOM_FACTOR;
                    }
                }

                // This is weird but look at the values.
                self.view.zoom = self.view.zoom.clamp(MIN_ZOOM, MAX_ZOOM);

                let center = resp.rect.center().to_vec2()
                    + if self.follow_selected_particle {
                        self.view.pos - self.particle_positions[self.selected_particle]
                    } else {
                        self.view.pos
                    } * self.view.zoom;

                if let Some(interact_pos) = ctx.input(|i| i.pointer.interact_pos()) {
                    if self.view.dragging {
                        let drag_delta = interact_pos - self.view.drag_start_pos;
                        self.view.pos =
                            self.view.drag_start_view_pos + drag_delta.to_vec2() / self.view.zoom;
                    }
                    if ctx.input(|i| i.pointer.button_down(PointerButton::Primary))
                        && resp.rect.contains(interact_pos)
                    {
                        if !self.view.dragging {
                            self.view.dragging = true;
                            self.view.drag_start_pos = interact_pos.to_vec2();
                            self.view.drag_start_view_pos = self.view.pos;
                        }
                    } else {
                        self.view.dragging = false;
                    }
                }

                let pos = Vec2::new(resp.rect.right() - 30., resp.rect.bottom() - 10.);
                paint.line_segment(
                    [
                        (pos - Vec2::new(10. * self.view.zoom, 0.)).to_pos2(),
                        pos.to_pos2(),
                    ],
                    Stroke::new(4., Color32::WHITE),
                );
                paint.text(
                    (pos + Vec2::new(10., 0.)).to_pos2(),
                    Align2::LEFT_CENTER,
                    "10",
                    FontId::monospace(14.),
                    Color32::WHITE,
                );

                for c in 0..CLASS_COUNT {
                    let class = &self.classes[c];

                    for p in 0..self.particle_counts[c] {
                        let pos =
                            (center + self.particle_positions[(c, p)] * self.view.zoom).to_pos2();
                        if paint.clip_rect().contains(pos) {
                            paint.circle_filled(
                                pos,
                                if (c, p) == self.selected_particle {
                                    PARTICLE_DIAMETER * 3.
                                } else {
                                    PARTICLE_DIAMETER
                                } * self.view.zoom,
                                class.color,
                            );
                        }
                    }
                }
            });

        ctx.request_repaint();
    }
}
