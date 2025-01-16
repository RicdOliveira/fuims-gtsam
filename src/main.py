from preprocess import load_data, process_dvl, process_imu, process_navigation, process_pressure
from gtsam_integration import setup_graph, optimize_graph

def main():
    # Caminhos para os ficheiros
    data_path = "../data/"
    dvl_file = f"{data_path}EVA_VMES_DVL.txt"
    imu_file = f"{data_path}EVA_VMES_IMU.txt"
    nav_file = f"{data_path}EVA_VMES_NAV.txt"
    pressure_file = f"{data_path}EVA_VMES_PRESSURE.txt"

    # Carregar e processar os dados
    dvl_data = load_data(dvl_file)
    imu_data = load_data(imu_file)
    nav_data = load_data(nav_file)
    pressure_data = load_data(pressure_file)

    dvl_timestamps, dvl_velocities = process_dvl(dvl_data)
    imu_timestamps, imu_accels, imu_gyros = process_imu(imu_data)
    nav_timestamps, nav_positions, nav_orientations = process_navigation(nav_data)
    pressure_timestamps, pressures = process_pressure(pressure_data)
    
    print("\nDados processados:")
    print(f"DVL Timestamps (amostra): {dvl_timestamps[:5]}")
    print(f"DVL Velocities (amostra): {dvl_velocities[:5]}")
    print(f"IMU Timestamps (amostra): {imu_timestamps[:5]}")
    print(f"IMU Accelerations (amostra): {imu_accels[:5]}")
    print(f"IMU Gyros (amostra): {imu_gyros[:5]}")
    print(f"NAV Timestamps (amostra): {nav_timestamps[:5]}")
    print(f"NAV Positions (amostra): {nav_positions[:5]}")
    print(f"NAV Orientations (amostra): {nav_orientations[:5]}")
    print(f"Pressure Timestamps (amostra): {pressure_timestamps[:5]}")
    print(f"Pressures (amostra): {pressures[:5]}")

    # Configurar grafo
    graph, initial_estimate = setup_graph(dvl_velocities, imu_accels, imu_gyros, nav_positions, nav_orientations)

    # Otimizar
    result = optimize_graph(graph, initial_estimate)

    # Salvar resultados
    for i in range(result.size()):
        print(f"Resultado para nó {i}: {result.atPose3(X(i))}")

if __name__ == "__main__":
    main()

