use eframe::egui::{Response, Sense, Ui, Vec2, Widget};

pub struct DirectionKnob<'a> {
    angle: &'a mut f32,
}

impl<'a> DirectionKnob<'a> {
    pub fn new(angle: &'a mut f32) -> Self {
        Self { angle }
    }
}

impl<'a> Widget for DirectionKnob<'a> {
    fn ui(self, ui: &mut Ui) -> Response {
        let (rect, mut response) = ui.allocate_exact_size(Vec2::new(100.0, 100.0), Sense::drag());

        if let Some(pos) = response.interact_pointer_pos() {
            if response.dragged() {
                *self.angle = -(pos - rect.center()).angle();
                response.mark_changed();
            }
        }

        let center = rect.center();
        let radius = rect.width() / 2.;
        let angle = *self.angle;

        ui.painter().circle(
            center,
            radius,
            ui.visuals().widgets.inactive.bg_fill,
            ui.visuals().widgets.inactive.bg_stroke,
        );
        ui.painter().arrow(
            center - Vec2::angled(-angle) * radius * 0.8,
            Vec2::angled(-angle) * radius * 0.8 * 2.,
            ui.visuals().widgets.active.fg_stroke,
        );

        response
    }
}
