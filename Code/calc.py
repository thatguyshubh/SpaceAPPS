import numpy as np
import backend

def run_pipeline():
    # Strip spaces and accidental quotes from dragged-and-dropped paths
    dir_input = input("Enter the directory path: ").strip().strip("'").strip('"')
    
    try:
        bands = backend.load_raster_data(dir_input)
    except FileNotFoundError as e:
        print(f"Error loading raster data: {e}")
        return
    
    indices = backend.calculate_indices(bands)
    ndvi = indices["ndvi"]
    ndwi = indices["ndwi"]
    ndmi = indices["ndmi"]
    rock = indices["rock_index"]  # Fixed key name to match backend.py

    # Filter for barren, dry terrain (changed 0.5 to 0.05 for arid soil)
    lunar = (ndvi < 0.12) & (ndwi < 0.0) & (ndmi < 0.05)
    usability = float(np.mean(lunar))

    # Fixed check: compare the float 'usability', not the 2D array 'lunar'
    if usability == 0:
        print("This terrain is completely unsuitable for a lunar base. It somehow scored a usability of 0.")
        return
    
    rock_score = float(np.mean(rock[lunar]))
    composition = float(np.clip(0.5 + 0.5 * rock_score, 0, 1))

    final_score = round(usability * composition * 100, 3)
    
    print(f"\nUsability:   {usability:.4f} ({usability * 100:.2f}%)")
    print(f"Composition: {composition:.4f}")
    print(f"Final score: {final_score} / 100")
    

    # Generate Best & Worst Points
    insights = backend.generate_site_insights(indices, usability, composition)

    print("\n[+] BEST POINTS:")
    for point in insights["best_points"]:
        print(f"  + {point}")

    print("\n[-] WORST POINTS:")
    for point in insights["worst_points"]:
        print(f"  - {point}")

    # Delegate plotting to backend.py instead of repeating matplotlib code
    choice = input("\nPlot result? (NDVI / ROCK / NDWI / MASK / NONE): ").strip().upper()
    if choice == "NDVI":
        backend.plot_layer(ndvi, "NDVI")
    elif choice == "ROCK":
        backend.plot_layer(rock, "ROCK")
    elif choice == "NDWI":
        backend.plot_layer(ndwi, "NDWI")
    elif choice == "MASK":
        backend.plot_layer(lunar, "MASK")

if __name__ == "__main__":
    run_pipeline()