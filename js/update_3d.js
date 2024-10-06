function(
    buffer_str,
) {
    let app = document.querySelector("gradio-app");
    app = app.shadowRoot ?? app;
    buffer_str = app.querySelector("#buffer textarea").value
    return [
        buffer_str,
    ]
}