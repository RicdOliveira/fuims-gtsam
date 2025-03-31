import gtsam
from gtsam import Pose3, Rot3, Point3
from gtsam.symbol_shorthand import X
from gtsam import NonlinearFactorGraph, Values, BetweenFactorPose3, noiseModel

def main():
    # Inicialização do gráfico e estimativas
    graph = NonlinearFactorGraph()
    initial_estimates = Values()

    # Modelos de ruído
    prior_noise = noiseModel.Diagonal.Sigmas([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    measurement_noise = noiseModel.Diagonal.Sigmas([0.2, 0.2, 0.2, 0.5, 0.5, 0.5])

    # Pose inicial (prior)
    initial_pose = Pose3(Rot3.RzRyRx(0, 0, 0), Point3(0, 0, 0))
    graph.add(gtsam.PriorFactorPose3(X(0), initial_pose, prior_noise))
    initial_estimates.insert(X(0), initial_pose)

    # Simulação de medições em tempo real (duas leituras por iteração)
    measurements = [
        (
            Pose3(Rot3.RzRyRx(0.1, 0.05, -0.02), Point3(1.0, 0.5, 0.2)),  # Leitura 1
            Pose3(Rot3.RzRyRx(0.15, 0.1, -0.01), Point3(1.1, 0.55, 0.25))  # Leitura 2
        ),
        (
            Pose3(Rot3.RzRyRx(0.2, -0.1, 0.1), Point3(2.0, 1.0, 0.5)),  # Leitura 1
            Pose3(Rot3.RzRyRx(0.18, -0.08, 0.12), Point3(2.1, 1.05, 0.55))  # Leitura 2
        ),
        (
            Pose3(Rot3.RzRyRx(0.15, 0.1, -0.05), Point3(3.0, 1.5, 0.8)),  # Leitura 1
            Pose3(Rot3.RzRyRx(0.16, 0.09, -0.06), Point3(3.1, 1.55, 0.85))  # Leitura 2
        )
    ]

    for i, (measurement1, measurement2) in enumerate(measurements):
        # Adiciona fatores para as duas medições conectando X(i) e X(i + 1)
        graph.add(BetweenFactorPose3(X(i), X(i + 1), measurement1, measurement_noise))
        graph.add(BetweenFactorPose3(X(i), X(i + 1), measurement2, measurement_noise))
        
        # Estimativa inicial para o próximo nó
        if i == 0:
            new_estimate = initial_pose.compose(measurement1)
        else:
            new_estimate = initial_estimates.atPose3(X(i)).compose(measurement1)
        initial_estimates.insert(X(i + 1), new_estimate)

        # Otimiza o gráfico
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates)
        result = optimizer.optimize()

        # Exibe a pose otimizada para o nó atual
        print(f"Pose otimizada para X({i + 1}):")
        print(result.atPose3(X(i + 1)))

        # Atualiza o initial_estimates para a próxima iteração
        initial_estimates = result

if __name__ == "__main__":
    main()
