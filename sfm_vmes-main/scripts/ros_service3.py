#!/usr/bin/env python3

import subprocess
import time

def launch_terminator_with_ros():
    # Define the commands to be run in each pane
    commands = [
        "roscore",
        "roslaunch sfm_node sfm_node.launch",
        "rqt",
        "rosbag play /home/andre/Documents/universidade/mestrado/1ano/LASMU/Bags/2024-11-20-14-39-32.bag"
        
    ]

    # Abre uma nova janela do Terminator com layout padrão 2x2
    subprocess.Popen(["terminator"])
    time.sleep(3)  # Aguarda o Terminator abrir completamente

    for i, cmd in enumerate(commands):
        print(f"Executando comando {i+1}: {cmd}")
        subprocess.run(["xdotool", "type", cmd])    # Digita o comando inteiro
        time.sleep(1)
        subprocess.run(["xdotool", "key", "Return"])  # Pressiona Enter para executar o comando
        time.sleep(3)  # Aguarda um pouco para garantir que o comando foi iniciado
        if i < len(commands) - 1:
            subprocess.run(["xdotool", "key", "Ctrl+Tab"])  # Muda para o próximo painel, exceto no último comando


if __name__ == "__main__":
    launch_terminator_with_ros()