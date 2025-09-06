// Use the Result type from the `anyhow` crate
use anyhow::Result;

use layoutparser_ort::models::{Detectron2Model, Detectron2PretrainedModels};
use pdfium_render::prelude::*;
use image::DynamicImage;


fn main() -> Result<()> {
    // Initialize Pdfium
    // This `?` will now work because `PdfiumError` can be converted into `anyhow::Error`.
    let bindings = Pdfium::bind_to_system_library()?;
    let pdfium = Pdfium::new(bindings);

    // Load PDF
    let doc = pdfium.load_pdf_from_file("/home/ubuntu/leptless/2021.eacl-main.75.pdf", None)?;

    // OPTIMIZATION: Load the model once, *before* the loop.
    let model = Detectron2Model::pretrained(Detectron2PretrainedModels::FASTER_RCNN_R_50_FPN_3X)?;

    for (i, page) in doc.pages().iter().enumerate() {
        // Render high-res bitmap
        let bitmap = page.render_with_config(
            &PdfRenderConfig::new()
                .set_target_width(2000)
                .set_maximum_height(2000)
                .rotate_if_landscape(PdfPageRenderRotation::None, false)
                .render_form_data(true),
        )?;
        let dyn_img: DynamicImage = bitmap.as_image();

        // This `?` already worked, but will now also convert `layoutparser_ort::Error`
        // into `anyhow::Error`.
        let predictions = model.predict(&dyn_img)?;

        println!("Predictions for page {}: {:?}", i + 1, predictions);
    }
    Ok(())
}