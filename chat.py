from file import EXAMPLES, model_inference
import gradio as gr

# Main application block
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 多模态智能聊天机器人", elem_classes="main-header")
    gr.ChatInterface(
            description="随时欢迎提问，还可以上传图片、视频提问哦！",
            examples=EXAMPLES,
            fill_height=False,
            fn=model_inference,
            multimodal=True,
            submit_btn="Submit",
            stop_btn="Stop", 
            # theme="soft",
            # title="Kai",
            type="messages",
    )

if __name__ == "__main__":
    demo.launch()
    