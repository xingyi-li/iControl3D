function(
    transform,
    render_output_state,
) {
    let app = document.querySelector("gradio-app");
    app = app.shadowRoot ?? app;
    transform = app.querySelector("#transform textarea").value

    return [
        transform,
        render_output_state,
    ]
}