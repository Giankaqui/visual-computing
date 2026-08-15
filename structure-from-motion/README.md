# Structure from Motion

Structure from motion incremental para cámaras calibradas, escrito sobre NumPy y
SciPy. La geometría está implementada desde cero: el solver mínimo de la matriz
esencial, la triangulación, la estimación de pose absoluta, la construcción de
tracks y el bundle adjuster están todos en este repositorio. OpenCV se usa solo
para decodificar imágenes y describir con SIFT.

![Reconstrucción de la escena sintética de referencia](docs/synthetic_demo.png)

## El pipeline

| Etapa | Módulo | Método |
| --- | --- | --- |
| Detección y descripción | [features.py](src/sfm/features.py) | SIFT, descriptores normalizados en L2 |
| Emparejamiento | [features.py](src/sfm/features.py) | Fuerza bruta por bloques, test de razón de Lowe, consistencia mutua |
| Geometría de dos vistas | [five_point.py](src/sfm/five_point.py), [epipolar.py](src/sfm/epipolar.py) | Solver esencial de cinco puntos dentro de MSAC con optimización local |
| Construcción de tracks | [tracks.py](src/sfm/tracks.py) | Union-find sobre los emparejamientos verificados, descartando componentes en conflicto |
| Triangulación | [triangulation.py](src/sfm/triangulation.py) | DLT multivista seguido de refinamiento Gauss-Newton |
| Registro | [pnp.py](src/sfm/pnp.py) | DLT normalizado dentro de RANSAC y luego Levenberg-Marquardt robusto |
| Refinamiento | [bundle.py](src/sfm/bundle.py) | Levenberg-Marquardt disperso con complemento de Schur y pérdida de Huber |
| Orquestación | [reconstruction.py](src/sfm/reconstruction.py) | Par semilla, registro incremental, refinamiento local y global intercalados |

## Notas de implementación

### Solver mínimo de cinco puntos

Cinco correspondencias entre vistas calibradas dejan un espacio nulo de dimensión
cuatro de la restricción epipolar, así que la matriz esencial se escribe como
`E = x E1 + y E2 + z E3 + E4`. Sustituyendo eso en las dos propiedades algebraicas
de una matriz esencial,

```
det(E) = 0        2 E Eᵀ E − traza(E Eᵀ) E = 0
```

salen diez polinomios cúbicos en tres incógnitas. Expresado en los veinte
monomios de grado como mucho tres, el sistema es una matriz `10 × 20` cuyo bloque
principal `10 × 10` es genéricamente invertible. Eliminarlo escribe cada monomio
cúbico en la base del anillo cociente `{x², xy, y², xz, yz, z², x, y, z, 1}`, que
es justo lo que hace falta para construir la matriz de multiplicación por `x` en
ese anillo; sus autovectores son los monomios de la base evaluados en las
soluciones.

El sistema polinómico se monta simbólicamente en tiempo de ejecución a partir de
la base del espacio nulo, en lugar de partir de tablas de coeficientes escritas a
mano, de modo que la construcción se puede leer y contrastar con la derivación.
Los tests verifican que toda matriz devuelta tiene valores singulares `(σ, σ, 0)`,
que satisface la restricción epipolar hasta precisión de máquina, y que la matriz
verdadera está entre las soluciones.

Frente al algoritmo de ocho puntos esto reduce a la mitad la muestra de RANSAC.
Con un 50 % de inliers el número esperado de hipótesis cae unos dos órdenes de
magnitud, y las restricciones de matriz esencial se imponen de forma exacta en
vez de restaurarse proyectando después una matriz general.

### Bundle adjustment

Las ecuaciones normales de una reconstrucción tienen la estructura por bloques

```
| U   W | | δ_c |   | g_c |
|       | |     | = |     |
| Wᵀ  V | | δ_p |   | g_p |
```

con un bloque `6 × 6` por cámara en `U`, uno `3 × 3` por punto en `V` y uno
`6 × 3` por observación en `W`. Como `V` es diagonal por bloques se invierte en
forma cerrada, y eliminar los puntos da el sistema reducido de cámaras
`(U − W V⁻¹ Wᵀ) δ_c = g_c − W V⁻¹ g_p`, cuyo tamaño solo depende del número de
vistas. Eso es lo que hace el paso asequible: para unos cientos de cámaras y cien
mil puntos el sistema denso queda fuera de alcance mientras que el reducido es
una resolución dispersa de unos pocos miles de incógnitas.

Tres detalles importan en la práctica.

* Las rotaciones se actualizan multiplicativamente, `R ← exp(skew(δ)) R`. El
  incremento es una carta local alrededor de la estimación actual, así que el
  jacobiano de un punto rotado es la matriz de producto vectorial `−skew(R X)` y
  nunca se roza una singularidad de la carta. El jacobiano se contrasta con
  diferencias centradas en los tests.
* Los residuos se reponderan con una pérdida de Huber, de modo que los
  emparejamientos erróneos que sobreviven doblan la solución una cantidad
  acotada. Apagar la pérdida degrada las poses de forma medible sobre datos
  contaminados, y los tests lo comprueban.
* Una reconstrucción libre tiene siete grados de libertad de gauge. Fijar una
  pose elimina seis; la libertad de escala que queda la absorbe el
  amortiguamiento de Levenberg-Marquardt, que mantiene el sistema reducido
  definido positivo. Por eso la estructura hay que alinearla con una similitud
  antes de compararla con la verdad.

El solver se valida contra `scipy.optimize.least_squares` corriendo
trust-region reflective sobre el mismo problema con un jacobiano disperso
diferenciado numéricamente. Los dos costes finales tienen que coincidir dentro
del dos por ciento.

### Estimación robusta

[ransac.py](src/sfm/ransac.py) es agnóstico al modelo: quien lo llama aporta un
solver mínimo y una función de residuos. Puntúa las hipótesis con el error
cuadrático truncado de MSAC en vez de con un conteo crudo de inliers, así que dos
modelos con el mismo conjunto de inliers se siguen ordenando por lo bien que lo
explican, y reajusta sobre el conjunto de inliers cada vez que mejora la mejor
puntuación. Ese paso de optimización local es lo que hace competitiva a una
muestra mínima frente a un ajuste no mínimo, y al agrandar el conjunto de inliers
también aprieta la cota adaptativa de iteraciones.

## Precisión en el banco de pruebas sintético

Doce cámaras sobre un arco de 90 grados observan 800 puntos repartidos en tres
paredes ortogonales. A cada proyección se le añade ruido gaussiano isótropo y a
una fracción de las observaciones se le añade un error grosero con desviación
típica de 40 píxeles. Los errores de estructura y pose se miden tras alinear la
reconstrucción con la verdad mediante una similitud; la escena mide unas 6
unidades de lado a lado.

| Ruido (px) | Outliers | Vistas | Puntos | RMSE (px) | Rotación, mediana (grados) | Centro, mediana | Estructura, mediana |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.0 | 0 % | 12/12 | 724 | 0.000 | 0.0000 | 0.00000 | 0.00000 |
| 0.0 | 15 % | 12/12 | 676 | 0.075 | 0.0043 | 0.00015 | 0.00045 |
| 0.5 | 0 % | 12/12 | 715 | 0.661 | 0.0195 | 0.00101 | 0.00271 |
| 0.5 | 15 % | 12/12 | 675 | 0.657 | 0.0074 | 0.00086 | 0.00233 |
| 1.0 | 0 % | 12/12 | 711 | 1.322 | 0.0478 | 0.00203 | 0.00614 |
| 1.0 | 15 % | 12/12 | 646 | 1.298 | 0.0325 | 0.00188 | 0.00529 |
| 1.5 | 0 % | 12/12 | 547 | 1.869 | 0.0745 | 0.00340 | 0.00930 |
| 1.5 | 15 % | 12/12 | 538 | 1.828 | 0.0867 | 0.00312 | 0.01126 |

Para reproducir una fila:

```bash
sfm demo --views 12 --points 800 --noise 0.5 --outliers 0.15 --seed 5
```

Dos observaciones. El RMSE residual sigue casi exactamente al ruido inyectado,
que es la firma de un estimador que no está absorbiendo el ruido de medida
dentro del modelo. Añadir outliers baja el número de puntos, porque los
emparejamientos contaminados destruyen los tracks que tocan, pero apenas mueve el
error de pose: eso son la pérdida robusta y la verificación geométrica haciendo
su trabajo.

## Uso

```bash
pip install -e structure-from-motion
```

Reconstruir una carpeta de imágenes:

```bash
sfm reconstruct ruta/a/imagenes --output out/ --fov 62
```

El directorio de salida recibe `points.ply`, un `cameras.json` con las poses
calibradas y una figura resumen. Los dos ficheros los consume directamente el
proyecto de [Gaussian splatting](../gaussian-splatting) de este repositorio, que
es el mismo relevo que hace un pipeline real entre un reconstructor disperso y un
renderizador.

Sin `--fov` la distancia focal se adivina a partir de un campo de visión
horizontal de 55 grados. Eso basta para arrancar una reconstrucción de capturas
informales, pero el bundle adjustment mantiene los intrínsecos fijos, así que una
suposición muy equivocada aparece como error sistemático que ninguna cantidad de
optimización elimina.

Correr en su lugar el banco de pruebas sintético, que no necesita datos de
entrada:

```bash
sfm demo --views 12 --points 800 --output out/
```

## Estructura

```
src/sfm/
  rotations.py       exponencial y logaritmo en SO(3), jacobianos locales
  camera.py          intrínsecos de pinhole, poses rígidas, proyección
  ransac.py          MSAC agnóstico al modelo con optimización local
  five_point.py      solver mínimo de la matriz esencial
  epipolar.py        solver de ocho puntos, error de Sampson, recuperación de pose
  triangulation.py   triangulación lineal y no lineal
  pnp.py             pose por DLT normalizado, refinamiento robusto
  tracks.py          tracks de características con union-find
  bundle.py          Levenberg-Marquardt disperso con complemento de Schur
  reconstruction.py  el bucle incremental
  metrics.py         alineamiento por similitud y métricas de error de pose
  io.py              serialización de PLY y cámaras
  synthetic.py       generador de escenas con verdad conocida
  visualize.py       figuras de resultados
  cli.py             interfaz de línea de comandos
tests/               47 tests, unos tres segundos
```

## Alcance

Los intrínsecos quedan fijos durante la optimización, así que la distorsión de
lente hay que quitarla antes y la distancia focal hay que conocerla o adivinarla
bien. El emparejamiento es exhaustivo y por tanto cuadrático en el número de
imágenes, que es la elección correcta hasta unas pocas docenas de vistas y la
equivocada más allá; colecciones mayores necesitan una etapa de recuperación de
imágenes que preseleccione pares. El registro usa un DLT de seis puntos en lugar
de un solver de tres, lo que cuesta iteraciones de RANSAC pero no es el cuello de
botella con las tasas de inliers que producen los tracks verificados.

## Referencias

* Nistér, *An Efficient Solution to the Five-Point Relative Pose Problem*, PAMI 2004.
* Stewénius, Engels y Nistér, *Recent Developments on Direct Relative Orientation*, ISPRS 2006.
* Hartley, *In Defense of the Eight-Point Algorithm*, PAMI 1997.
* Hartley y Zisserman, *Multiple View Geometry in Computer Vision*, 2ª edición, 2004.
* Triggs, McLauchlan, Hartley y Fitzgibbon, *Bundle Adjustment: A Modern Synthesis*, 1999.
* Chum, Matas y Kittler, *Locally Optimized RANSAC*, DAGM 2003.
* Torr y Zisserman, *MLESAC: A New Robust Estimator*, CVIU 2000.
* Schönberger y Frahm, *Structure-from-Motion Revisited*, CVPR 2016.
