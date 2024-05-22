# 字体管理
# 创建人：曾逸夫
# 创建时间：2022-05-01


import os
import sys
from pathlib import Path

import wget
from rich.console import Console


FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]

# 中文、英文、俄语、西班牙语、阿拉伯语、韩语
fonts_list = ["SimSun.ttf", "TimesNewRoman.ttf", "malgun.ttf"]  # 字体列表
fonts_suffix = ["ttc", "ttf", "otf"]  # 字体后缀

data_url_dict = {
    "SimSun.ttf": "https://gitee.com/CV_Lab/gradio_yolov5_det/attach_files/1053539/download/SimSun.ttf",
    "TimesNewRoman.ttf": "https://gitee.com/CV_Lab/gradio_yolov5_det/attach_files/1053537/download/TimesNewRoman.ttf",
    "malgun.ttf": "https://gitee.com/CV_Lab/gradio_yolov5_det/attach_files/1053538/download/malgun.ttf",
}

console = Console()


# 创建字体库
def add_fronts(font_diff):
    global font_name

    for k, v in data_url_dict.items():
        if k in font_diff:
            font_name = v.split("/")[-1]  # 字体名称
            Path(ROOT / "fonts").mkdir(parents=True, exist_ok=True)  # 创建目录

            file_path = f"{ROOT}/fonts/{font_name}"  # 字体路径

            try:
                # 下载字体文件
                wget.download(v, file_path)
            except Exception as e:
                console.print("[bold red]路径错误！程序结束！[/bold red]")
                print(e)
                sys.exit()
            else:
                print()
                console.print(
                    f"{font_name} [bold green]字体文件下载完成！[/bold green] 已保存至：{file_path}"
                )


# 判断字体文件
def is_fonts(fonts_dir):
    if os.path.isdir(fonts_dir):
        # 如果字体库存在
        f_list = os.listdir(fonts_dir)  # 本地字体库

        font_diff = list(set(fonts_list).difference(set(f_list)))

        if font_diff != []:
            # 字体不存在
            console.print("[bold red]字体不存在，正在加载。。。[/bold red]")
            add_fronts(font_diff)  # 创建字体库
        else:
            console.print(f"{fonts_list}[bold green]字体已存在！[/bold green]")
    else:
        # 字体库不存在，创建字体库
        console.print("[bold red]字体库不存在，正在创建。。。[/bold red]")
        add_fronts(fonts_list)  # 创建字体库
