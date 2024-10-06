var camera_postprocessing=function(){
    if(!window.my_observe_camera)
    {
        console.log("setup camera postprocessing here");
        window.my_observe_camera = new MutationObserver(function (event) {
            console.log(event);
            let app=document.querySelector("gradio-app");
            app=app.shadowRoot??app;
            let frame=app.querySelector("#sdinfframe").contentWindow;
            frame.postMessage(["update_buffer", ""], "*");
        });
        var app=document.querySelector("gradio-app");
        app=app.shadowRoot??app;
        window.my_observe_camera_target=app.querySelector("#render_output span");
        window.my_observe_camera.observe(window.my_observe_camera_target, {
            attributes: false, 
            subtree: true,
            childList: true, 
            characterData: true
        });
    }
};
camera_postprocessing();