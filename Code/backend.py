from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import rasterio


def load_raster_data(directory_path: str) -> dict:
    data_dir = Path(directory_path)
    band_files = {
        "b12": data_dir / "B12RAW.tiff",
        "b11": data_dir / "B11RAW.tiff",
        "b04": data_dir / "B04RAW.tiff",
        "b08": data_dir / "B08RAW.tiff",
        "b03": data_dir / "B03RAW.tiff",
    }

    missing = [name for name, p in band_files.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files: {missing}")

    with rasterio.open(band_files["b04"]) as r_src:
        red = r_src.read(1).astype(np.float32)
    with rasterio.open(band_files["b08"]) as nir_src:
        nir = nir_src.read(1).astype(np.float32)
    with rasterio.open(band_files["b03"]) as g_src:
        green = g_src.read(1).astype(np.float32)
    with rasterio.open(band_files["b11"]) as swir1_src:
        swir1 = swir1_src.read(1).astype(np.float32)
    with rasterio.open(band_files["b12"]) as swir2_src:
        swir2 = swir2_src.read(1).astype(np.float32)

    return {
        "red": red,
        "nir": nir,
        "green": green,
        "swir1": swir1,
        "swir2": swir2,
    }


def calculate_indices(bands: dict) -> dict:
    red, nir, green = bands["red"], bands["nir"], bands["green"]
    swir1, swir2 = bands["swir1"], bands["swir2"]

    denom_ndvi = nir + red
    ndvi = np.where(
        denom_ndvi == 0, 0, (nir - red) / np.where(denom_ndvi == 0, 1.0, denom_ndvi)
    )

    denom_ndwi = green + nir
    ndwi = np.where(
        denom_ndwi == 0,
        0,
        (green - nir) / np.where(denom_ndwi == 0, 1.0, denom_ndwi),
    )

    denom_ndmi = nir + swir1
    ndmi = np.where(
        denom_ndmi == 0,
        0,
        (nir - swir1) / np.where(denom_ndmi == 0, 1.0, denom_ndmi),
    )

    denom_rock = swir1 + nir
    rock_index = np.where(
        denom_rock == 0,
        0,
        (swir1 - nir) / np.where(denom_rock == 0, 1.0, denom_rock),
    )

    silicate_ratio = np.where(
        swir1 == 0, 0, swir2 / np.where(swir1 == 0, 1.0, swir1)
    )

    # Ferric Oxide (Iron) Index: High Red relative to Green highlights oxidized Martian-style regolith
    denom_iron = red + green
    ferric_index = np.where(
        denom_iron == 0,
        0,
        (red - green) / np.where(denom_iron == 0, 1.0, denom_iron),
    )

    return {
        "ndvi": ndvi,
        "ndwi": ndwi,
        "ndmi": ndmi,
        "rock_index": rock_index,
        "silicate_ratio": silicate_ratio,
        "ferric_index": ferric_index,
    }


def plot_layer(data: np.ndarray, layer_type: str, target_body: str = "LUNAR"):
    """
    Standardized visualizer for rasters and calculated indices.
    """
    plt.figure(figsize=(9, 7))
    layer = layer_type.upper()
    body_label = target_body.capitalize()

    if layer == "NDVI":
        plt.imshow(data, cmap="RdYlGn", vmin=-0.1, vmax=0.5)
        plt.colorbar(label="NDVI (Low = Barren/Rock, High = Flora)")
        plt.title("NDVI Vegetation Index")
    elif layer in ("NDRI", "ROCK"):
        plt.imshow(data, cmap="magma")
        plt.colorbar(label="NDRI (Higher = Mafic/Basaltic)")
        plt.title("Rock / Mineral Index (SWIR-1 vs NIR)")
    elif layer == "FERRIC":
        plt.imshow(data, cmap="OrRd")
        plt.colorbar(label="Ferric Index (Higher = Iron-Oxide Rich)")
        plt.title("Ferric Oxide / Hematite Index (Red vs Green)")
    elif layer == "NDWI":
        plt.imshow(data, cmap="Blues")
        plt.colorbar(label="NDWI (Water/Moisture)")
        plt.title("NDWI Surface Water Index")
    elif layer == "MASK":
        plt.imshow(data, cmap="gray")
        plt.colorbar(label="Barren Ground (1 = True, 0 = False)")
        plt.title(f"Filtered {body_label} Candidate Mask")
    else:
        plt.imshow(data, cmap="viridis")
        plt.colorbar(label="Intensity")
        plt.title(f"{layer_type} Map")

    plt.tight_layout()
    plt.show()


def generate_site_insights(
    indices: dict, usability: float, composition: float, target_body: str = "LUNAR"
) -> dict:
    """
    Analyzes spectral indices to generate target-specific Advantages and Disadvantages.
    """
    best_points = []
    worst_points = []
    target = target_body.upper()

    mean_ndvi = float(np.mean(indices["ndvi"]))
    mean_ndwi = float(np.mean(indices["ndwi"]))
    mean_ndmi = float(np.mean(indices["ndmi"]))
    mean_silicate = float(np.mean(indices["silicate_ratio"]))
    mean_ferric = float(np.mean(indices["ferric_index"]))

    # 1. Evaluate Usable Barren Area (Shared)
    if usability >= 0.75:
        best_points.append(
            f"Expansive unobstructed barren terrain ({usability*100:.1f}% usable area for rover/EVA traversal)."
        )
    elif usability >= 0.40:
        best_points.append(
            f"Moderate contiguous barren zones ({usability*100:.1f}% usable area)."
        )
    else:
        worst_points.append(
            f"Severely limited barren ground ({usability*100:.1f}% usable); heavy environmental interference."
        )

    # 2. Evaluate Vegetation (NDVI) (Shared)
    if mean_ndvi < 0.12:
        best_points.append(
            f"Near-zero biological/flora interference (Mean NDVI: {mean_ndvi:.3f})."
        )
    elif mean_ndvi > 0.22:
        worst_points.append(
            f"High vegetation coverage obscures regolith mechanics (Mean NDVI: {mean_ndvi:.3f})."
        )

    # 3. Standing Water Check (Shared)
    if mean_ndwi > 0.15:
        worst_points.append(
            "Significant standing liquid water or riparian drainage channels present."
        )

    # 4. Target-Specific Geological & Hydration Insights
    if target == "LUNAR":
        # Lunar requires extreme desiccation
        if mean_ndmi < 0.0:
            best_points.append(
                f"Highly desiccated surface soil ideal for anhydrous lunar regolith simulation (Mean NDMI: {mean_ndmi:.3f})."
            )
        elif mean_ndmi > 0.08:
            worst_points.append(
                f"Elevated soil moisture compromises dry lunar vacuum-simulant fidelity (Mean NDMI: {mean_ndmi:.3f})."
            )

        # Lunar favors fresh, unweathered basaltic mare rock
        if composition >= 0.58:
            best_points.append(
                f"Strong mafic/basaltic mineral signature matching lunar mare (Affinity: {composition:.2f})."
            )
        elif composition < 0.48:
            worst_points.append(
                f"Weak basaltic signature; surface dominated by quartz sand, clay, or felsic rock (Affinity: {composition:.2f})."
            )

        # Clay alteration is a disadvantage for the Moon
        if mean_silicate >= 0.85:
            best_points.append(
                "Low clay alteration; unweathered ferrous silicate rock exposed."
            )
        elif mean_silicate < 0.70:
            worst_points.append(
                "High chemical weathering or hydrous clay presence detected in SWIR-2 (unrealistic for lunar regolith)."
            )

    elif target == "MARTIAN":
        # Mars analogs welcome ferric iron oxides (hematite/goethite)
        if mean_ferric >= 0.12:
            best_points.append(
                f"Strong ferric oxide (iron-rich) spectral signature matching Martian regolith (Ferric Index: {mean_ferric:.3f})."
            )
        elif mean_ferric < 0.03:
            worst_points.append(
                f"Low iron-oxide coloration; lacks characteristic Martian ferric dust/soil signature (Ferric Index: {mean_ferric:.3f})."
            )

        # Mars missions specifically target ancient aqueous clays/phyllosilicates & basalt
        if 0.60 <= mean_silicate <= 0.85:
            best_points.append(
                f"Phyllosilicate/clay alteration signatures detected (SWIR ratio: {mean_silicate:.2f}) — high-value astrobiology/paleolake analog."
            )
        elif mean_silicate > 0.85 and composition >= 0.55:
            best_points.append(
                "Unweathered basaltic plains signature matching Martian volcanic provinces (e.g., Elysium/Tharsis)."
            )
        elif mean_silicate < 0.55:
            worst_points.append(
                "Excessive terrestrial humid weathering/leaching reduces Martian fidelity."
            )

        # Mars allows bound hydration/gypsum/salts, only penalizing wet mud/swamp
        if -0.05 <= mean_ndmi <= 0.12:
            best_points.append(
                f"Arid evaporite/hydrated-mineral regime consistent with Martian subsurface/playa analogs (Mean NDMI: {mean_ndmi:.3f})."
            )
        elif mean_ndmi > 0.18:
            worst_points.append(
                f"Excessive liquid soil saturation unrepresentative of hyper-arid Martian conditions (Mean NDMI: {mean_ndmi:.3f})."
            )

    # Fallbacks
    if not best_points:
        best_points.append(
            "Baseline geological stability; no extreme surface anomalies."
        )
    if not worst_points:
        worst_points.append(
            "No major spectral disqualifiers detected across loaded bands."
        )

    return {
        "best_points": best_points,
        "worst_points": worst_points,
    } 