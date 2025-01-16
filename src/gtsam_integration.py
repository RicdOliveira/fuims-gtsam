import gtsam
from gtsam.symbol_shorthand import X, V, B  # Pose3, Velocity, Bias

def setup_graph(dvl_velocities, imu_accels, imu_gyros, nav_positions, nav_orientations):
    """Configura o grafo do GTSAM com os sensores."""
    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # Prior inicial
    prior_pose = gtsam.Pose3()  # Posição inicial na origem
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas([0.1] * 6)  # Exemplo de ruído
    graph.add(gtsam.PriorFactorPose3(X(0), prior_pose, prior_noise))
    initial_estimate.insert(X(0), prior_pose)

    # Adicionar fatores DVL
    for i, velocity in enumerate(dvl_velocities):
        velocity_noise = gtsam.noiseModel.Isotropic.Sigma(3, 0.1)
        graph.add(gtsam.PriorFactorVector(V(i), velocity, velocity_noise))
        initial_estimate.insert(V(i), gtsam.Point3(velocity))

    # Fatores de IMU podem ser adicionados aqui com PreintegratedImuMeasurements.

    # Outros fatores conforme necessário...

    return graph, initial_estimate

def optimize_graph(graph, initial_estimate):
    """Otimiza o grafo e retorna os resultados."""
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()
    return result

