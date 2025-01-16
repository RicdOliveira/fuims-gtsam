import pandas as pd

def load_data(file_path):
    """Carrega os dados ignorando comentários."""
    return pd.read_csv(file_path, comment='%', delimiter=',')

def process_dvl(dvl_data):
    """Extrai tempo e velocidades do DVL."""
    timestamps = dvl_data['field.time_stamp'].values
    velocities = dvl_data[['field.velocity_xyzz0', 'field.velocity_xyzz1', 'field.velocity_xyzz2']].values
    return timestamps, velocities

def process_imu(imu_data):
    """Extrai tempo, aceleração e giroscópio do IMU."""
    timestamps = imu_data['field.time_stamp'].values
    accelerations = imu_data[['field.acc0', 'field.acc1', 'field.acc2']].values
    gyros = imu_data[['field.gyro0', 'field.gyro1', 'field.gyro2']].values
    return timestamps, accelerations, gyros

def process_navigation(nav_data):
    """Extrai posição e orientação dos dados de navegação."""
    timestamps = nav_data['field.header.stamp'].values
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
    timestamps = pressure_data['field.time_stamp'].values
    pressures = pressure_data['field.pressure'].values
    return timestamps, pressures

