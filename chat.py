from file import model_inference
import gradio as gr


EXAMPLES = [
    [
        {
            "text": "你好",
        }
    ],
    [
        {
            "text": "自我介绍一下吧",
        }
    ],
    [
        {
            "text": "lcm是什么？",
        }
    ],
    [
        {
            "text": "lcm是什么？",
            "files": ["example_files/lcm.pdf"]
        }
    ],
    [
        {
            "text": "分析一下这张图片",
            "files": ["example_files/1.jpg"]
        }
    ],
    [
        {
            "text": "视频中的人物是谁？",
            "files": ["example_files/1.gif"]
        }
    ],
    [
        {
            "text": "这是游戏还是现实？",
            "files": ["example_files/1.mp4"]
        }
    ],

]


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
    