import numpy as np
import backend


def run_pipeline():
    # 1. Target Body Selector
    while True:
        target_input = (
            input("Select analog target [1: Lunar / 2: Martian]: ")
            .strip()
            .upper()
        )
        if target_input in ("1", "L", "LUNAR", "MOON"):
            target_body = "LUNAR"
            break
        elif target_input in ("2", "M", "MARTIAN", "MARS"):
            target_body = "MARTIAN"
            break
        print("Invalid selection. Please enter 'Lunar' (1) or 'Martian' (2).")

    # 2. Load Data
    dir_input = (
        input("Enter the directory path: ").strip().strip("'").strip('"')
    )

    try:
        bands = backend.load_raster_data(dir_input)
    except FileNotFoundError as e:
        print(f"Error loading raster data: {e}")
        return

    indices = backend.calculate_indices(bands)
    ndvi = indices["ndvi"]
    ndwi = indices["ndwi"]
    ndmi = indices["ndmi"]
    rock = indices["rock_index"]
    ferric = indices["ferric_index"]

    # 3. Target-Specific Terrain Mask & Composition Scoring
    if target_body == "LUNAR":
        # Strict hyper-arid, barren, non-vegetated mask
        mask = (ndvi < 0.12) & (ndwi < 0.0) & (ndmi < 0.05)
        usability = float(np.mean(mask))

        if usability == 0:
            print(
                "This terrain is completely unsuitable for a Lunar analog. Usability scored 0."
            )
            return

        rock_score = float(np.mean(rock[mask]))
        composition = float(np.clip(0.5 + 0.5 * rock_score, 0, 1))

    else:  # MARTIAN
        # Allows slightly higher bound hydration (clays/sulfates) while enforcing barrenness
        mask = (ndvi < 0.14) & (ndwi < 0.05) & (ndmi < 0.15)
        usability = float(np.mean(mask))

        if usability == 0:
            print(
                "This terrain is completely unsuitable for a Martian analog. Usability scored 0."
            )
            return

        # Blend basaltic rock signature (60%) + ferric iron-oxide enrichment (40%)
        rock_score = float(np.mean(rock[mask]))
        ferric_score = float(np.mean(ferric[mask]))
        basalt_norm = np.clip(0.5 + 0.5 * rock_score, 0, 1)
        ferric_norm = np.clip(0.5 + 1.5 * ferric_score, 0, 1)
        composition = float(0.6 * basalt_norm + 0.4 * ferric_norm)

    final_score = round(usability * composition * 100, 3)

    print(f"\n=== {target_body} ANALOG EVALUATION ===")
    print(f"Usability:   {usability:.4f} ({usability * 100:.2f}%)")
    print(f"Composition: {composition:.4f}")
    print(f"Final score: {final_score} / 100")

    # 4. Generate Target-Specific Advantages & Disadvantages
    insights = backend.generate_site_insights(
        indices, usability, composition, target_body=target_body
    )

    print(f"\n[+] {target_body} ADVANTAGES:")
    for point in insights["best_points"]:
        print(f"  + {point}")

    print(f"\n[-] {target_body} DISADVANTAGES:")
    for point in insights["worst_points"]:
        print(f"  - {point}")

    # 5. Plotting
    choice = (
        input("\nPlot result? (NDVI / ROCK / FERRIC / NDWI / MASK / NONE): ")
        .strip()
        .upper()
    )
    if choice == "NDVI":
        backend.plot_layer(ndvi, "NDVI", target_body)
    elif choice == "ROCK":
        backend.plot_layer(rock, "ROCK", target_body)
    elif choice == "FERRIC":
        backend.plot_layer(ferric, "FERRIC", target_body)
    elif choice == "NDWI":
        backend.plot_layer(ndwi, "NDWI", target_body)
    elif choice == "MASK":
        backend.plot_layer(mask, "MASK", target_body)


if __name__ == "__main__":
    run_pipeline()
