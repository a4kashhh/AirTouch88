# Hardware Design & Electrical Analysis

## 1. Schematic Topology

```
HIGH-SIDE ROW DRIVER (Rows 0–7):
     +5V Rail
        │
        ├──[ 10 kΩ Pull-up ]──┐
        │                     │
      Source (S)              │
   ┌──┴─────────┐             │
   │ IRF9540N   ├── Gate (G) ─┴── Collector (C)
   │  (P-FET)   │                 ┌──┴─────────┐
   └──┬─────────┘                 │   BC547    ├── Base (B) ──[ 1 kΩ ]── ESP32 GPIO
      Drain (D)                   │   (NPN)    │
        │                         └──┬─────────┘
        ▼                            │ Emitter (E)
   Matrix Row Anode (0..7)           ▼
                                    GND
─────────────────────────────────────────────────────────────────────────────
LOW-SIDE COLUMN DRIVER (Columns 0–7):
   Matrix Column Cathode (0..7)
        │
        └──[ 220 Ω Resistor ]
                  │
                Drain (D)
             ┌──┴─────────┐
             │   2N7000   ├── Gate (G) ──[ 100 Ω ]── ESP32 GPIO
             │  (N-FET)   │      │
             └──┬─────────┘   [100 kΩ Pull-down]
                │                │
              Source (S)         ▼
                │               GND
                ▼
               GND
```

## 2. Pin Mapping Table (ESP32-C3)

| Matrix Line | Function | ESP32-C3 GPIO | Interface Driver | Driver Pins | Matrix Line |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Row 0** | Anode Line 0 | **GPIO 0** | 1 kΩ $\to$ BC547 $\to$ IRF9540N | Gate $\to$ C; Source $\to$ +5V; Drain $\to$ Row 0 | Anodes (0,0)..(0,7) |
| **Row 1** | Anode Line 1 | **GPIO 1** | 1 kΩ $\to$ BC547 $\to$ IRF9540N | Gate $\to$ C; Source $\to$ +5V; Drain $\to$ Row 1 | Anodes (1,0)..(1,7) |
| **Row 2** | Anode Line 2 | **GPIO 2** | 2.2 kΩ $\to$ BC547 $\to$ IRF9540N | Gate $\to$ C; Source $\to$ +5V; Drain $\to$ Row 2 | Anodes (2,0)..(2,7) |
| **Row 3** | Anode Line 3 | **GPIO 3** | 1 kΩ $\to$ BC547 $\to$ IRF9540N | Gate $\to$ C; Source $\to$ +5V; Drain $\to$ Row 3 | Anodes (3,0)..(3,7) |
| **Row 4** | Anode Line 4 | **GPIO 4** | 1 kΩ $\to$ BC547 $\to$ IRF9540N | Gate $\to$ C; Source $\to$ +5V; Drain $\to$ Row 4 | Anodes (4,0)..(4,7) |
| **Row 5** | Anode Line 5 | **GPIO 5** | 1 kΩ $\to$ BC547 $\to$ IRF9540N | Gate $\to$ C; Source $\to$ +5V; Drain $\to$ Row 5 | Anodes (5,0)..(5,7) |
| **Row 6** | Anode Line 6 | **GPIO 6** | 1 kΩ $\to$ BC547 $\to$ IRF9540N | Gate $\to$ C; Source $\to$ +5V; Drain $\to$ Row 6 | Anodes (6,0)..(6,7) |
| **Row 7** | Anode Line 7 | **GPIO 7** | 1 kΩ $\to$ BC547 $\to$ IRF9540N | Gate $\to$ C; Source $\to$ +5V; Drain $\to$ Row 7 | Anodes (7,0)..(7,7) |
| **Col 0** | Cathode Line 0 | **GPIO 8** | 100 Ω $\to$ 2N7000 (100k pull-down) | Source $\to$ GND; Drain $\to$ 220 Ω | Cathodes (0,0)..(7,0) |
| **Col 1** | Cathode Line 1 | **GPIO 10** | 100 Ω $\to$ 2N7000 (100k pull-down) | Source $\to$ GND; Drain $\to$ 220 Ω | Cathodes (0,1)..(7,1) |
| **Col 2** | Cathode Line 2 | **GPIO 18** | 100 Ω $\to$ 2N7000 (100k pull-down) | Source $\to$ GND; Drain $\to$ 220 Ω | Cathodes (0,2)..(7,2) |
| **Col 3** | Cathode Line 3 | **GPIO 19** | 100 Ω $\to$ 2N7000 (100k pull-down) | Source $\to$ GND; Drain $\to$ 220 Ω | Cathodes (0,3)..(7,3) |
| **Col 4** | Cathode Line 4 | **GPIO 20** | 100 Ω $\to$ 2N7000 (100k pull-down) | Source $\to$ GND; Drain $\to$ 220 Ω | Cathodes (0,4)..(7,4) |
| **Col 5** | Cathode Line 5 | **GPIO 21** | 100 Ω $\to$ 2N7000 (100k pull-down) | Source $\to$ GND; Drain $\to$ 220 Ω | Cathodes (0,5)..(7,5) |
| **Col 6** | Cathode Line 6 | **GPIO 9** | 100 Ω $\to$ 2N7000 (100k pull-down) | Source $\to$ GND; Drain $\to$ 220 Ω | Cathodes (0,6)..(7,6) |
| **Col 7** | Cathode Line 7 | **GPIO 11** | 100 Ω $\to$ 2N7000 (100k pull-down) | Source $\to$ GND; Drain $\to$ 220 Ω | Cathodes (0,7)..(7,7) |

## 3. Current & Voltage Derivations

* **Supply Voltage**: $V_{CC} = +5.0\,\text{V}$
* **Red LED Forward Drop**: $V_F \approx 1.9\,\text{V}$
* **MOSFET Drops**: $V_{DS(P-FET)} \approx 0.05\,\text{V}$, $V_{DS(N-FET)} \approx 0.10\,\text{V}$
* **Peak Branch Current**:
  $$I_{peak} = \frac{5.0 - 1.9 - 0.15}{220} \approx 13.4\,\text{mA}$$
* **Average Time-Averaged Current (1/8 TDM Duty Cycle)**:
  $$I_{avg} = 13.4\,\text{mA} \times \frac{1}{8} \approx 1.68\,\text{mA}$$
* **Dead-Time Blanking**: $15\,\mu\text{s}$ blanking period mitigates $C_{iss} \approx 1300\,\text{pF}$ gate discharge through the 10 kΩ pull-up, eliminating ghosting completely.
