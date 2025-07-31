import numpy as np
from datetime import datetime
from sklearn.utils.estimator_checks import check_estimator
from ps.ps1 import EstimadorOLS
from mlfin.printing import print_validation_results, log_info, log_error, setup_logging
from sklearn.datasets import make_regression


"""
Módulo de test para la clase EstimadorOLS. Tipos:
- test_type = 1: Test de la clase EstimadorOLS (con constante)
- test_type = 2: Test de la clase EstimadorOLS (sin constante)
- test_type = 3: Test de la clase EstimadorOLS (con diferentes shapes)

"""

test_type = 1  # Cambiar a 2 o 3 según el tipo de test que se quiera correr

if __name__ == "__main__":
    try:
        # Setup logging
        setup_logging()

        log_info("="*50)
        log_info(f"Realizando test del tipo: {test_type}")
        log_info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Se setea el seed
        seed = 1234
        np.random.seed(seed)
        log_info(f"Seed configurado: {seed}")

        # Se generan los datos
        X, y, original_coefs = make_regression(
            n_samples=1000,
            n_features=10,
            bias=5,
            noise=10,
            random_state=seed,
            coef=True
        )

        if test_type == 1:
            log_info("Corriendo el test de la clase EstimadorOLS")
            model = EstimadorOLS()

        elif test_type == 2:
            log_info("Corriendo el test de la clase EstimadorOLS sin constante")
            model = EstimadorOLS(fit_intercept=False)

        elif test_type == 3:
            log_info("Corriendo el test de la clase EstimadorOLS con diferentes shapes")

            # A y le saco una fila
            y = y[:-1]
            log_info(f"Modificando y - Nueva longitud: {len(y)} (se removió una muestra)")

            model = EstimadorOLS()

        # Se fittea el modelo y se muestran los resultados
        model.fit(X, y)
        np.set_printoptions(precision=8)

        log_info("Resultado de Cross-Validation:")
        print_validation_results(model, X, y)

        log_info("Original coefs:")
        log_info(f"{original_coefs}")

        log_info("Fitted coefs:")
        if test_type == 2:  # Sin constante
            log_info(f"Coeficientes: {model.coef_}")
        else:  # Con constante
            log_info(f"Constante: {model.intercept_:.6f}")
            log_info(f"Coeficientes: {model.coef_}")

    except Exception as e:
        log_error(f"Error durante la ejecución: {e}")
        exit(1)

    # Por último, se chequean los resultados de los common checks
    try:
        results = check_estimator(model)
        log_info("Se pasaron correctamente todos los checks.")
    except Exception as e:
        log_error(f"Error al generar los checks: {e}")