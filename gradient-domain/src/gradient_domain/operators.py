"""Operadores diferenciales discretos sobre una malla regular.

Toda aplicación de este paquete se reduce al mismo enunciado: dado un campo de
gradiente objetivo ``v`` que no tiene por qué ser integrable, encontrar la imagen
cuyo gradiente más se le acerca en el sentido de mínimos cuadrados.  Las
ecuaciones normales de ese problema son la ecuación de Poisson

.. math::

    \\nabla^2 u = \\nabla \\cdot v,

así que los operadores de abajo son los únicos que hacen falta: un gradiente por
diferencias hacia delante para construir campos guía, su adjunto negativo como
divergencia, y el laplaciano de cinco puntos que resulta de componerlos.

Usar el par adjunto en lugar de dos plantillas elegidas por separado es lo que
hace simétrico al problema discreto.  Un sistema simétrico es lo que permite
aplicar gradiente conjugado y multigrid, y es la razón de que la divergencia use
diferencias hacia atrás cuando el gradiente usa las de hacia delante.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

__all__ = ["gradient", "divergence", "laplacian", "sparse_laplacian", "fold_boundary"]


def gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Diferencias hacia delante en ambos ejes.

    La última fila y la última columna se ponen a cero, que es la condición de
    Neumann que hace que el adjunto del operador sea exactamente
    :func:`divergence`.

    Parameters
    ----------
    image : ndarray, shape (h, w) o (h, w, c)

    Returns
    -------
    gx, gy : ndarray
        Derivadas horizontal y vertical, con la misma forma que la entrada.
    """
    gx = np.zeros_like(image, dtype=float)
    gy = np.zeros_like(image, dtype=float)
    gx[:, :-1] = image[:, 1:] - image[:, :-1]
    gy[:-1, :] = image[1:, :] - image[:-1, :]
    return gx, gy


def divergence(gx: np.ndarray, gy: np.ndarray) -> np.ndarray:
    """Adjunto negativo de :func:`gradient`.

    La última fila y la última columna de la entrada se ignoran, porque el
    gradiente nunca escribe ahí y por tanto el adjunto no debe leer de ahí.  Que
    la relación sea exacta para campos arbitrarios, y no solo para los que salen
    de :func:`gradient`, es lo que mantiene simétrica la composición cuando un
    campo guía se construye con una operación que no preserva esa estructura.

    Parameters
    ----------
    gx, gy : ndarray, shape (h, w) o (h, w, c)

    Returns
    -------
    ndarray
        Misma forma que las entradas; ``divergence(*gradient(u))`` es el
        laplaciano de cinco puntos de ``u`` bajo condiciones de Neumann
        homogéneas.
    """
    dx = np.zeros_like(gx, dtype=float)
    dy = np.zeros_like(gy, dtype=float)
    if gx.shape[1] > 1:
        dx[:, 0] = gx[:, 0]
        dx[:, 1:-1] = gx[:, 1:-1] - gx[:, :-2]
        dx[:, -1] = -gx[:, -2]
    if gy.shape[0] > 1:
        dy[0, :] = gy[0, :]
        dy[1:-1, :] = gy[1:-1, :] - gy[:-2, :]
        dy[-1, :] = -gy[-2, :]
    return dx + dy


def laplacian(u: np.ndarray, spacing: float = 1.0) -> np.ndarray:
    """Laplaciano de cinco puntos con condiciones de Dirichlet homogéneas.

    Los valores fuera del array se tratan como cero, así que este es el operador
    del sistema interior que produce :func:`fold_boundary`.

    Parameters
    ----------
    u : ndarray, shape (m, n) o (m, n, c)
    spacing : float
        Paso de malla ``h``; el operador escala como ``1 / h ** 2``.

    Returns
    -------
    ndarray
        Misma forma que ``u``.
    """
    padded = np.pad(u, ((1, 1), (1, 1)) + ((0, 0),) * (u.ndim - 2))
    neighbours = (
        padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]
    )
    return (neighbours - 4.0 * u) / (spacing * spacing)


def sparse_laplacian(shape: tuple[int, int], spacing: float = 1.0) -> sp.csr_matrix:
    """Ensambla como matriz dispersa el laplaciano de cinco puntos de :func:`laplacian`.

    Parameters
    ----------
    shape : tuple of int
        Tamaño de la malla interior ``(m, n)``.
    spacing : float

    Returns
    -------
    scipy.sparse.csr_matrix, shape (m * n, m * n)
        Simétrica definida negativa, en orden por filas (C).
    """
    rows, cols = shape

    def second_difference(size: int) -> sp.dia_matrix:
        return sp.diags(
            [np.ones(size - 1), np.full(size, -2.0), np.ones(size - 1)], [-1, 0, 1], format="csr"
        )

    identity_rows = sp.identity(rows, format="csr")
    identity_cols = sp.identity(cols, format="csr")
    operator = sp.kron(second_difference(rows), identity_cols) + sp.kron(
        identity_rows, second_difference(cols)
    )
    return (operator / (spacing * spacing)).tocsr()


def fold_boundary(
    right_hand_side: np.ndarray, boundary: np.ndarray, spacing: float = 1.0
) -> np.ndarray:
    """Traslada los valores de Dirichlet conocidos del operador al término independiente.

    Las incógnitas son el interior de ``boundary``; su anillo de un píxel guarda
    los valores prescritos.  Cada píxel interior contiguo al anillo pierde un
    vecino de la plantilla, y el término correspondiente pasa al término
    independiente con el signo cambiado.

    Parameters
    ----------
    right_hand_side : ndarray, shape (h - 2, w - 2) o (h - 2, w - 2, c)
        Divergencia del campo guía sobre el interior.
    boundary : ndarray, shape (h, w) o (h, w, c)
        Imagen cuyo borde aporta los valores de contorno.
    spacing : float
        Paso de malla, el mismo que use el operador.

    Returns
    -------
    ndarray
        Misma forma que ``right_hand_side``.
    """
    folded = np.array(right_hand_side, dtype=float, copy=True)
    weight = 1.0 / (spacing * spacing)
    folded[0] -= weight * boundary[0, 1:-1]
    folded[-1] -= weight * boundary[-1, 1:-1]
    folded[:, 0] -= weight * boundary[1:-1, 0]
    folded[:, -1] -= weight * boundary[1:-1, -1]
    return folded
