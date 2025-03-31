import pandas as pd
import numpy as np


def load_data(file_path):
    """Carrega os dados ignorando comentários."""
    return pd.read_csv(file_path, comment='%', delimiter=',')


def process_dvl(dvl_data):
    """Extrai tempo e velocidades do DVL."""
    timestamps = dvl_data['time'].values
    velocities = dvl_data[['field.velocity_xyzz0',
                           'field.velocity_xyzz1', 'field.velocity_xyzz2']].values
    return timestamps, velocities


def process_imu(imu_data):
    """Extrai tempo, aceleração e giroscópio do IMU."""
    timestamps = imu_data['time'].values
    accelerations = imu_data[['field.acc0', 'field.acc1', 'field.acc2']].values
    gyros = imu_data[['field.gyro0', 'field.gyro1', 'field.gyro2']].values
    return timestamps, accelerations, gyros


def process_navigation(nav_data):
    """Extrai posição e orientação dos dados de navegação."""
    timestamps = nav_data['time'].values
    positions = nav_data[['field.pose.pose.position.x',
                          'field.pose.pose.position.y',
                          'field.pose.pose.position.z']].values
    orientations = nav_data[['field.pose.pose.orientation.x',
                             'field.pose.pose.orientation.y',
                             'field.pose.pose.orientation.z',
                             'field.pose.pose.orientation.w']].values
    return timestamps, positions, orientations


def process_pressure(pressure_data):
    """Extrai tempo e pressão do sensor de pressão."""
    timestamps = pressure_data['time'].values
    pressures = pressure_data['field.pressure'].values
    return timestamps, pressures


def calculate_sampling_rate(timestamps):
    """Calcula a taxa de amostragem (em Hz) com base nos timestamps."""
    intervals = np.diff(timestamps)  # Intervalos de tempo consecutivos
    avg_interval = np.mean(intervals)  # Intervalo médio
    # Taxa de amostragem (convertido de nanossegundos para segundos)
    sampling_rate = 1 / (avg_interval / 1e9)
    # Retorna a taxa de amostragem e o intervalo médio
    return sampling_rate, avg_interval


def synchronize_data(nav_timestamps, nav_positions, nav_orientations, pressure_timestamps, pressures):
    """
    Sincroniza os dados de navegação e pressão a cada 0.2 segundos.
    """

    # Definir o intervalo de tempo para a sincronização (0.2s em nanossegundos)
    sync_interval_ns = int(0.2 * 1e9)

    # Determinar o intervalo de tempo comum
    start_time = max(nav_timestamps[0], pressure_timestamps[0])
    end_time = min(nav_timestamps[-1], pressure_timestamps[-1])

    # Criar timestamps sincronizados
    synced_timestamps = np.arange(start_time, end_time, sync_interval_ns)

    synced_nav_positions = []
    synced_nav_orientations = []
    synced_pressures = []

    for t in synced_timestamps:
        # Encontrar o índice mais próximo para navegação
        nav_index = np.argmin(np.abs(nav_timestamps - t))
        synced_nav_positions.append(nav_positions[nav_index])
        synced_nav_orientations.append(nav_orientations[nav_index])

        # Encontrar o índice mais próximo para pressão
        pressure_index = np.argmin(np.abs(pressure_timestamps - t))
        synced_pressures.append(pressures[pressure_index])

    return synced_nav_positions, synced_nav_orientations, synced_pressures
