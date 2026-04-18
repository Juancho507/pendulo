# Pendulo — CartPole con instrucciones en lenguaje natural

Sistema que convierte instrucciones en español o inglés en comportamiento de un agente RL que balancea un péndulo invertido.

---

## Requisitos

- Docker y Docker Compose instalados
- Cualquier navegador web (Chrome, Firefox, Edge...)

---

## Uso rápido

### 1. Construir la imagen
```bash
docker compose build
```

### 2. Entrenar el agente
```bash
docker compose run train
```

Con instrucción personalizada:
```bash
docker compose run train python main.py "minimiza movimientos bruscos y prioriza estabilidad"
```

Instrucciones de ejemplo:
- `"mantén el equilibrio el mayor tiempo posible"`
- `"minimiza movimientos bruscos"`
- `"prioriza estabilidad"`
- `"maintain balance as long as possible"`

### 3. Visualizar en el navegador
```bash
docker compose up visualize
```

Luego abre tu navegador en:
```
http://localhost:6080/vnc.html
```

Y verás el agente balanceando el péndulo en tiempo real.

---

## Estructura del proyecto

```
pendulo/
├── main.py               # Entrena y evalúa el agente
├── visualize.py          # Visualización gráfica
├── start_vnc.sh          # Levanta Xvfb + x11vnc + noVNC
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── model/                # Modelo guardado (se crea al entrenar)
├── nlp/
│   └── interpreter.py    # Texto → parámetros de comportamiento
└── rl/
    ├── env.py            # Entorno con recompensa customizada
    ├── trainer.py        # Entrenamiento PPO
    └── evaluator.py      # Evaluación del modelo
```

---

## Cómo funciona

1. **NLP** — `interpreter.py` detecta palabras clave en la instrucción (español/inglés) y genera 3 parámetros:
   - `stability` → penaliza la inclinación del palo
   - `smoothness` → penaliza movimientos bruscos del carrito
   - `duration` → premia sobrevivir más tiempo

2. **RL** — PPO (Proximal Policy Optimization) entrena en CartPole-v1 usando esos parámetros para moldear la función de recompensa

3. **Visualización** — El contenedor levanta un display virtual (Xvfb), lo expone via VNC, y noVNC lo sirve en el navegador por el puerto 6080
