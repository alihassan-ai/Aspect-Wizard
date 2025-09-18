from ui.ui import build_ui

if __name__ == "__main__":
    demo = build_ui()
    demo.queue(concurrency_count=1, max_size=32)  
    demo.launch(
        server_name="0.0.0.0",
        server_port=7803,
        share=False,  # flip to True if public link is needed
        inbrowser=True
    )
