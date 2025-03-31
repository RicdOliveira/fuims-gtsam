import gtsam
import numpy as np
from gtsam import Rot3
from gtsam.symbol_shorthand import X
from src.preprocess import load_data, process_dvl, process_imu, process_navigation, process_pressure, synchronize_data


def quaternion_to_rot3_no_library(quat):
    """
    Converte um quaternião [w, x, y, z] diretamente em uma matriz de rotação.
    """
    x, y, z, w = quat
    # Fórmula para a matriz de rotação 3x3
    R = np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2)]
    ])
    return Rot3(R)  # Converte a matriz de rotação para Rot3 do GTSAM


def get_nav_pos(navegation_poses, index):
    """
    Retorna as matrizes de rotação e translação num momento da navegação.
    """
    quat = navegation_poses["orientations"][index][0], navegation_poses["orientations"][index][
        1], navegation_poses["orientations"][index][2], navegation_poses["orientations"][index][3]

    position = navegation_poses["positions"][index][0], navegation_poses["positions"][index][
        1], navegation_poses["positions"][index][2]

    orientation = quaternion_to_rot3_no_library(quat)

    return position, orientation


import gtsam
from gtsam.symbol_shorthand import X

def estimate_trajectory_gtsam(navegation_poses, pressures):
    # Inicializa o gráfico de fatores e as estimativas iniciais
    graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()

    # Modelos de ruído
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.1] * 6)  # Ruído para fator prior
    measurement_noise = gtsam.noiseModel.Diagonal.Sigmas([0.2, 0.2, 0.2, 0.5, 0.5, 0.5])  # Ruído para medições

    # Adiciona fator prior na primeira posição (âncora do grafo)
    pos1, rot1 = get_nav_pos(navegation_poses, 0)
    initial_pose = gtsam.Pose3(rot1, pos1)

    graph.add(gtsam.PriorFactorPose3(X(0), initial_pose, prior_noise))
    initial_estimates.insert(X(0), initial_pose)

    # Itera sobre todas as medições
    for i in range(1, len(navegation_poses["positions"])):
        # Obtém a posição absoluta da navegação e da câmara
        nav_pos, nav_rot = get_nav_pos(navegation_poses, i)
        # No momento, é igual à navegação
        cam_pos, cam_rot = get_nav_pos(navegation_poses, i)

        absolute_nav_measurement = gtsam.Pose3(nav_rot, nav_pos)
        absolute_cam_measurement = gtsam.Pose3(cam_rot, cam_pos)

        # Adiciona um fator absoluto da navegação a cada 5 medições
        if i % 5 == 0:
            graph.add(gtsam.PriorFactorPose3(X(i), absolute_nav_measurement, measurement_noise))

        # Adiciona um fator absoluto da câmara a cada 5 medições
        if i % 5 == 0:
            graph.add(gtsam.PriorFactorPose3(X(i), absolute_cam_measurement, measurement_noise))

        # Obtém a posição anterior
        prev_nav_pos, prev_nav_rot = get_nav_pos(navegation_poses, i - 1)
        prev_pose = gtsam.Pose3(prev_nav_rot, prev_nav_pos)

        # Calcula a transformação relativa
        relative_measurement = prev_pose.between(absolute_nav_measurement)
        graph.add(gtsam.BetweenFactorPose3(X(i - 1), X(i), relative_measurement, measurement_noise))

        # Insere a estimativa inicial no grafo
        initial_estimates.insert(X(i), absolute_nav_measurement)

        # Exibe a estimativa inicial
        print(f"Estimativa inicial para X({i}): {initial_estimates.atPose3(X(i))}")

        # Otimiza parcialmente a cada iteração
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates)
        temp_result = optimizer.optimize()

        # Exibe a estimativa otimizada parcial
        print(f"Estimativa otimizada parcial para X({i}): {temp_result.atPose3(X(i))}")

    # Realiza a otimização final
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates)
    result = optimizer.optimize()

    # Exibe as poses otimizadas finais
    for i in range(len(navegation_poses["positions"])):
        print(f"Pose otimizada final para X({i}): {result.atPose3(X(i))}")

    return result


def main():
    # File paths
    data_path = "../data/"
    dvl_file = f"{data_path}EVA_VMES_DVL.txt"
    imu_file = f"{data_path}EVA_VMES_IMU.txt"
    nav_file = f"{data_path}EVA_VMES_NAV.txt"
    pressure_file = f"{data_path}EVA_VMES_PRESSURE.txt"

    # Load and process data
    dvl_data = load_data(dvl_file)
    imu_data = load_data(imu_file)
    nav_data = load_data(nav_file)
    pressure_data = load_data(pressure_file)

    dvl_timestamps, dvl_velocities = process_dvl(dvl_data)
    imu_timestamps, imu_accels, imu_gyros = process_imu(imu_data)
    nav_timestamps, nav_positions, nav_orientations = process_navigation(
        nav_data)
    pressure_timestamps, pressures = process_pressure(pressure_data)

    # Suponha que você tenha os dados carregados em suas variáveis
    synced_nav_positions, synced_nav_orientations, synced_pressures = synchronize_data(
        nav_timestamps, nav_positions, nav_orientations, pressure_timestamps, pressures)
    synced_nav = {"positions": synced_nav_positions,
                  "orientations": synced_nav_orientations}

    trajectory = estimate_trajectory_gtsam(synced_nav, synced_pressures)


if __name__ == "__main__":
    main()
