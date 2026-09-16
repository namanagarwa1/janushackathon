import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# PARAGLIDER PATH FINDING
# ============================================================

# -----------------------------
# Parameters
# -----------------------------

# From the paper / model
V = 6.2                  # forward velocity [m/s]
sink_rate = 1.92         # descent rate [m/s]

# Turning radius reported in the paper
R_paper = 102.0          # [m]

# Smaller radius allowed during final approach
R_final = 12.0           # [m]

# Maximum turn rate
omega_max = V / R_final

# Simulation parameters
dt = 0.05
initial_height = 700.0

# ------------------------------------------------------------
# Random starting position
# ------------------------------------------------------------

np.random.seed(42)

x0 = np.random.uniform(-500, 500)
y0 = np.random.uniform(-500, 500)

# Initially point toward target
psi0 = np.arctan2(-y0, -x0)

print("====================================")
print("PARAGLIDER SIMULATION")
print("====================================")

print(f"Starting X      : {x0:.2f} m")
print(f"Starting Y      : {y0:.2f} m")
print(f"Starting height : {initial_height:.2f} m")


# ============================================================
# FUNCTIONS
# ============================================================

def wrap_angle(angle):
    """
    Convert angle to [-pi, pi]
    """
    return np.arctan2(
        np.sin(angle),
        np.cos(angle)
    )


# ============================================================
# GUIDANCE LAW
# ============================================================

def guidance(x, y, z, psi):

    r = np.sqrt(x**2 + y**2)

    # --------------------------------------------------------
    # Required altitude to fly directly to target
    #
    # glide ratio = horizontal velocity / sink velocity
    # --------------------------------------------------------

    required_height = r * sink_rate / V

    # ========================================================
    # FINAL APPROACH
    # ========================================================

    # Once there is only enough altitude to reach the target,
    # stop circling and fly directly toward (0,0).
    #
    # The +10 m gives the controller some margin.
    # ========================================================

    if z < required_height + 10:

        desired_heading = np.arctan2(
            -y,
            -x
        )

        phase = "FINAL APPROACH"

    # ========================================================
    # SPIRAL DESCENT
    # ========================================================

    else:

        theta = np.arctan2(y, x)

        # Desired spiral radius decreases with altitude.
        #
        # At high altitude:
        #       ~100 m
        #
        # Near target:
        #       ~20-30 m
        #

        radius_desired = (
            R_final
            + (R_paper - R_final)
            * (z / initial_height)
        )

        radius_desired = np.clip(
            radius_desired,
            R_final,
            R_paper
        )

        # ----------------------------------------------------
        # Tangential direction
        # ----------------------------------------------------

        tangent_heading = theta + np.pi / 2

        # ----------------------------------------------------
        # Radial correction
        #
        # If we are outside desired radius:
        #       turn inward
        #
        # If inside:
        #       turn outward
        # ----------------------------------------------------

        radial_error = r - radius_desired

        correction = np.arctan2(
            radial_error,
            25.0
        )

        desired_heading = (
            tangent_heading
            + correction
        )

        phase = "SPIRAL DESCENT"

    return wrap_angle(desired_heading), phase


# ============================================================
# SIMULATION
# ============================================================

def simulate():

    # Maximum flight time
    max_time = initial_height / sink_rate

    N = int(max_time / dt) + 1

    # State arrays
    x = np.zeros(N)
    y = np.zeros(N)
    z = np.zeros(N)
    psi = np.zeros(N)

    # Initial conditions
    x[0] = x0
    y[0] = y0
    z[0] = initial_height
    psi[0] = psi0

    final_phase_reached = False

    for i in range(N - 1):

        # ----------------------------------------------------
        # Stop if we reach ground
        # ----------------------------------------------------

        if z[i] <= 0:

            x[i:] = x[i]
            y[i:] = y[i]
            z[i:] = 0
            psi[i:] = psi[i]

            break

        # ----------------------------------------------------
        # Get guidance command
        # ----------------------------------------------------

        desired_heading, phase = guidance(
            x[i],
            y[i],
            z[i],
            psi[i]
        )

        if phase == "FINAL APPROACH":
            final_phase_reached = True

        # ----------------------------------------------------
        # Heading error
        # ----------------------------------------------------

        heading_error = wrap_angle(
            desired_heading - psi[i]
        )

        # ----------------------------------------------------
        # Proportional heading controller
        # ----------------------------------------------------

        K_heading = 6.0

        turn_rate = (
            K_heading * heading_error
        )

        # ----------------------------------------------------
        # Limit turn rate
        # ----------------------------------------------------

        turn_rate = np.clip(
            turn_rate,
            -omega_max,
            omega_max
        )

        # ----------------------------------------------------
        # Update heading
        # ----------------------------------------------------

        psi[i + 1] = (
            psi[i]
            + turn_rate * dt
        )

        # ----------------------------------------------------
        # Horizontal dynamics
        # ----------------------------------------------------

        x[i + 1] = (
            x[i]
            + V
            * np.cos(psi[i + 1])
            * dt
        )

        y[i + 1] = (
            y[i]
            + V
            * np.sin(psi[i + 1])
            * dt
        )

        # ----------------------------------------------------
        # Vertical dynamics
        # ----------------------------------------------------

        z[i + 1] = max(
            0,
            z[i]
            - sink_rate * dt
        )

    # --------------------------------------------------------
    # Find actual landing point
    # --------------------------------------------------------

    landing_index = np.where(z <= 0)[0]

    if len(landing_index) > 0:
        landing_index = landing_index[0]

        x = x[:landing_index + 1]
        y = y[:landing_index + 1]
        z = z[:landing_index + 1]
        psi = psi[:landing_index + 1]

    return x, y, z, psi, final_phase_reached


# ============================================================
# RUN
# ============================================================

x, y, z, psi, final_phase = simulate()


# ============================================================
# LANDING ERROR
# ============================================================

landing_x = x[-1]
landing_y = y[-1]
landing_z = z[-1]

horizontal_error = np.sqrt(
    landing_x**2 +
    landing_y**2
)

total_error = np.sqrt(
    landing_x**2 +
    landing_y**2 +
    landing_z**2
)

print("\n====================================")
print("LANDING RESULTS")
print("====================================")

print(f"Landing X       : {landing_x:.2f} m")
print(f"Landing Y       : {landing_y:.2f} m")
print(f"Landing Z       : {landing_z:.2f} m")

print(
    f"Horizontal error: "
    f"{horizontal_error:.2f} m"
)

print(
    f"Total error     : "
    f"{total_error:.2f} m"
)

print(
    f"Final approach  : "
    f"{final_phase}"
)


# ============================================================
# 3D PLOT
# ============================================================

fig = plt.figure(figsize=(11, 9))

ax = fig.add_subplot(
    111,
    projection="3d"
)

# ------------------------------------------------------------
# Flight path
# ------------------------------------------------------------

ax.plot(
    x,
    y,
    z,
    linewidth=2.5,
    label="Paraglider flight path"
)

# ------------------------------------------------------------
# Starting point
# ------------------------------------------------------------

ax.scatter(
    x[0],
    y[0],
    z[0],
    s=120,
    marker="o",
    label="Start"
)

# ------------------------------------------------------------
# Target
# ------------------------------------------------------------

ax.scatter(
    0,
    0,
    0,
    s=300,
    marker="*",
    label="Target (0,0,0)"
)

# ------------------------------------------------------------
# Vertical reference line
# ------------------------------------------------------------

ax.plot(
    [0, 0],
    [0, 0],
    [0, initial_height],
    linestyle="--",
    linewidth=1
)

# ------------------------------------------------------------
# Labels
# ------------------------------------------------------------

ax.set_xlabel("X position (m)")
ax.set_ylabel("Y position (m)")
ax.set_zlabel("Height (m)")

ax.set_title(
    "3D Paraglider Path Finding"
)

ax.legend()

plt.tight_layout()
plt.show()


# ============================================================
# TOP VIEW
# ============================================================

plt.figure(figsize=(9, 9))

plt.plot(
    x,
    y,
    linewidth=2.5,
    label="Flight path"
)

plt.scatter(
    x[0],
    y[0],
    s=120,
    label="Start"
)

plt.scatter(
    0,
    0,
    s=300,
    marker="*",
    label="Target"
)

# Paper turning radius
circle = plt.Circle(
    (0, 0),
    R_paper,
    fill=False,
    linestyle="--",
    linewidth=1,
    label="102 m turning radius"
)

plt.gca().add_patch(circle)

plt.xlabel("X position (m)")
plt.ylabel("Y position (m)")

plt.title(
    "Horizontal Paraglider Path"
)

plt.axis("equal")
plt.grid(True)
plt.legend()

plt.show()