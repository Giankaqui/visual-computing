# Proyectos de Computación Visual

Tres proyectos autocontenidos que cubren el camino de las fotografías a una
escena 3D renderizada, más la maquinaria de dominio del gradiente que comparten
la composición de imágenes y el mapeo tonal. Cada uno es un paquete instalable
por separado con sus propios tests, benchmarks y documentación.

| Proyecto | Qué hace | Técnica central |
| --- | --- | --- |
| [structure-from-motion](structure-from-motion) | Recupera poses de cámara y una nube dispersa a partir de imágenes | Solver mínimo de cinco puntos, bundle adjustment con complemento de Schur |
| [gaussian-splatting](gaussian-splatting) | Ajusta una escena 3D a imágenes con pose conocida y renderiza vistas nuevas | Proyección EWA, rasterizador diferenciable por tiles, control adaptativo de densidad |
| [gradient-domain](gradient-domain) | Compone, aplana y mapea tonalmente imágenes | Integración de Poisson con un solver multigrid geométrico |

Una [demo interactiva](interactive-demo) mueve los tres desde el navegador, con
los sliders conectados al mismo código de librería que llama la línea de
comandos.

```bash
python interactive-demo/app.py
```

Los dos primeros proyectos se componen. Structure from motion escribe
`cameras.json` y `points.ply`; el entrenador de splatting lee exactamente esos
ficheros y se inicializa con ellos, que es el mismo relevo que hace un pipeline
de producción entre un reconstructor disperso y un renderizador.

```bash
sfm reconstruct fotos/ --output escena/
gsplat train --scene escena/ --images fotos/ --output modelo/
```

Esa cadena pide fotografías reales. La escena procedural que trae el proyecto de
splatting es un banco de pruebas de síntesis de vistas, no de emparejamiento de
características, y reconstruirla a partir de sus propios renders solo registra
una fracción de las vistas, por [motivos que son de las imágenes](gaussian-splatting#alcance)
y no del pipeline.

## Qué está implementado aquí en vez de llamado

El objeto de estos proyectos son los algoritmos, así que las partes que cargan
con las ideas están escritas y no delegadas:

* el solver de pose relativa de cinco puntos, incluida la construcción simbólica
  de las diez restricciones cúbicas y la resolución por autovalores de la matriz
  de acción;
* bundle adjustment disperso con complemento de Schur, parametrización local en
  `SO(3)` y pérdida robusta;
* la proyección EWA de gaussianas 3D anisótropas y un rasterizador diferenciable
  por tiles, con checkpointing de activaciones para acotar su memoria;
* el control de densidad que decide cuántas primitivas necesita una escena,
  incluida la cirugía sobre el estado del optimizador que mantiene a Adam
  coherente al hacerlo;
* un ciclo V multigrid geométrico con suavizado Gauss-Seidel rojo-negro y
  transferencias de malla adjuntas.

Se usa código de terceros donde no es el objeto de estudio: OpenCV para decodificar
imágenes y describir con SIFT, SciPy para factorización dispersa y transformadas,
PyTorch para diferenciación automática y operaciones sobre arrays.

## Cómo correrlo todo

Cada proyecto se instala de forma independiente y no necesita descargar datos;
todas las demostraciones generan sus propias entradas.

```bash
python -m venv .venv && source .venv/bin/activate

pip install -e structure-from-motion && sfm demo --output out/sfm
pip install -e gaussian-splatting   && gsplat train --scene synthetic --output out/gsplat
pip install -e gradient-domain      && gradient-domain demo --output out/gradient
```

Los tests, por proyecto:

```bash
cd structure-from-motion && pytest -q
```

## Resultados de un vistazo

**Structure from motion.** Doce vistas de una escena de 800 puntos, medio píxel
de ruido de medida y un 15 % de outliers groseros: todas las vistas registradas,
error de rotación mediano de 0.007 grados, error de estructura del 0.04 % del
diámetro de la escena.
[Detalles](structure-from-motion#precisión-en-el-banco-de-pruebas-sintético)

**Dominio del gradiente.** A un megapíxel, el gradiente conjugado necesita 1866
iteraciones para llegar a un residuo relativo de `1e-8` y el gradiente conjugado
precondicionado con multigrid necesita 7, un factor de 53 en tiempo de reloj.
[Detalles](gradient-domain#escalado)

**Gaussian splatting.** Cuarenta vistas de una escena trazada por rayos a
200 x 150, partiendo de diez mil primitivas colocadas al azar y sin nube de
puntos: 24.23 dB y 0.907 de SSIM en vistas retenidas, medio decibelio por debajo
de las de entrenamiento. [Detalles](gaussian-splatting#resultados)

## Convenios compartidos entre los proyectos

Las cámaras van de mundo a cámara, `x_camera = R x_world + t`, con el eje óptico
en `+z` y `+y` apuntando hacia abajo en la imagen; esto coincide con OpenCV y
COLMAP, así que las poses viajan entre proyectos sin cambio de base. Las imágenes
son arrays en coma flotante en `[0, 1]` con los canales al final. Toda rutina
aleatoria acepta una semilla y es reproducible.

## Nota sobre el idioma

La documentación, los comentarios y la interfaz están en español. Los
identificadores del código (funciones, clases, variables) y los encabezados de
sección de los docstrings se mantienen en inglés, que es el convenio habitual y
lo que preserva la correspondencia con la literatura: `essential matrix`,
`bundle adjustment`, `multigrid`.

## Licencia

MIT. Ver [LICENSE](LICENSE).
