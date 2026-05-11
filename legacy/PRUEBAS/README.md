# Legado BTC-SHSE

Este directorio documenta el experimento histórico SHSE del repositorio original.

SHSE intentaba proyectar ventanas temporales BTC sobre una esfera de Fibonacci y usar k-NN angular como estructura relacional. La hipótesis produjo una lección importante: una geometría uniforme en S² no preserva automáticamente vecindad temporal. En particular, los vecinos angulares de una esfera de Fibonacci pueden ser temporalmente arbitrarios, por lo que usar valores reconstruidos como sustituto de las features temporales puede destruir señal predictiva.

Lecciones preservadas:

- No asumir una variedad antes de diagnosticar los datos.
- Separar geometría fija de estado latente entrenable.
- Tratar SHSE como enriquecimiento relacional, no como sustituto ciego de las features originales.
- Medir reconstrucción, suavidad y uso latente antes de usar una geometría para predicción.

El nuevo proyecto ESO generaliza esta intuición: explora varias variedades candidatas y registra qué geometría explica mejor la estructura interna de los datos.

Los archivos originales permanecen temporalmente en la raíz durante la transición para evitar pérdida accidental de código. A medida que ESO gane cobertura, se moverán o eliminarán del punto de entrada activo.
