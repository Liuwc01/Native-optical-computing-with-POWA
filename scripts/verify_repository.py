"""Fast integrity checks for the committed source data."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_checksum_list(checksum_list: Path) -> tuple[int, set[str]]:
    """Verify one checksum list and return its referenced relative paths."""

    checked_files = 0
    referenced: set[str] = set()
    for line_number, line in enumerate(
        checksum_list.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        parts = line.split(maxsplit=1)
        if len(parts) != 2:
            raise AssertionError(
                f"invalid checksum entry in {checksum_list}:{line_number}"
            )
        expected, relative_name = parts
        relative_name = relative_name.strip().lstrip("*")
        candidate = checksum_list.parent / relative_name
        if not candidate.is_file():
            raise AssertionError(f"missing checksummed file: {candidate}")
        observed = sha256(candidate)
        if observed != expected.lower():
            raise AssertionError(
                f"checksum mismatch for {candidate}: {observed} != {expected}"
            )
        checked_files += 1
        referenced.add(relative_name)
    return checked_files, referenced


def verify_figure_package_checksums() -> tuple[int, int]:
    """Verify every figure-data checksum list and parse every manifest."""

    figure_root = ROOT / "data" / "figure_source_data"
    checked_files = 0
    checksum_lists = sorted(figure_root.rglob("SHA256SUMS.txt"))
    for checksum_list in checksum_lists:
        checked, _ = verify_checksum_list(checksum_list)
        checked_files += checked

    manifests = sorted(figure_root.rglob("manifest.json"))
    for manifest in manifests:
        json.loads(manifest.read_text(encoding="utf-8"))
    return checked_files, len(manifests)


def verify_ovgg_sample() -> tuple[int, float]:
    """Verify the bundled O-VGG codebook and evaluation records."""

    sample = ROOT / "examples" / "ovgg_cifar10_sample"
    checked, referenced = verify_checksum_list(sample / "SHA256SUMS.txt")
    manifest = json.loads((sample / "manifest.json").read_text(encoding="utf-8"))
    manifest_paths = {entry["path"] for entry in manifest["files"]}
    if referenced != manifest_paths:
        raise AssertionError("O-VGG manifest and checksum file lists differ")
    for entry in manifest["files"]:
        candidate = sample / entry["path"]
        if candidate.stat().st_size != entry["bytes"]:
            raise AssertionError(f"unexpected file size: {candidate}")
        if sha256(candidate) != entry["sha256"]:
            raise AssertionError(f"O-VGG manifest hash mismatch: {candidate}")

    with np.load(sample / "ovgg_cifar10_k101_codebook.npz") as archive:
        codebook = np.asarray(archive["codebook"])
        response = np.asarray(archive["response_normalized"])
        weights = np.asarray(archive["weights_standardized_x0p2"])
        wavelength = np.asarray(archive["wavelength_nm"])
    if codebook.shape != (8, 101) or weights.shape != (8, 101):
        raise AssertionError("unexpected O-VGG codebook shape")
    if not np.array_equal(codebook, response.astype(np.float32)):
        raise AssertionError("normalized O-VGG response arrays differ")
    if not np.array_equal(wavelength, np.linspace(1545.0, 1555.0, 101)):
        raise AssertionError("unexpected O-VGG sample wavelength grid")

    matlab = loadmat(sample / "ovgg_cifar10_k101_codebook.mat")
    if not np.array_equal(matlab["response_normalized"], response):
        raise AssertionError("MAT and NPZ normalized responses differ")
    if not np.array_equal(matlab["weights_standardized_x0p2"], weights):
        raise AssertionError("MAT and NPZ processed weights differ")

    metrics = json.loads((sample / "test_metrics.json").read_text(encoding="utf-8"))
    if metrics["correct"] != 8_860 or metrics["total"] != 10_000:
        raise AssertionError("unexpected O-VGG sample metrics")
    return checked, float(metrics["accuracy"])


def main() -> None:
    metadata_path = ROOT / "data" / "metadata" / "wpu_spectrum.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source = ROOT / metadata["source_file"]
    if sha256(source) != metadata["source_sha256"]:
        raise AssertionError("raw WPU spectrum hash does not match metadata")

    with h5py.File(source, "r") as handle:
        variable = metadata["source_matlab_variable"]
        if variable not in handle:
            raise AssertionError(f"missing MATLAB variable: {variable}")
        observed_shape = list(handle[variable].shape)
    if observed_shape != metadata["hdf5_array_shape"]:
        raise AssertionError(
            f"source array shape changed: {observed_shape} != "
            f"{metadata['hdf5_array_shape']}"
        )

    figure_directory = (
        ROOT
        / "data"
        / "figure_source_data"
        / "fig2c_figS1_wpu_broadband_response"
    )
    figure_manifest = json.loads(
        (figure_directory / "manifest.json").read_text(encoding="utf-8")
    )
    figure_file = figure_directory / figure_manifest["derived_file"]["path"]
    if sha256(figure_file) != figure_manifest["derived_file"]["sha256"]:
        raise AssertionError("figure source-data hash does not match its manifest")
    figure_data = loadmat(figure_file)
    wavelength_nm = np.asarray(figure_data["wavelength_nm"]).reshape(-1)
    response_db = np.asarray(figure_data["response_db"])
    source_indices = np.asarray(
        figure_data["source_sample_index_1based"]
    ).reshape(-1)
    expected_wavelength = np.linspace(1500.0, 1620.0, 12_001)
    expected_indices = np.arange(1, 2_400_002, 200)
    if not np.array_equal(wavelength_nm, expected_wavelength):
        raise AssertionError("unexpected wavelength grid in figure source data")
    if not np.array_equal(source_indices, expected_indices):
        raise AssertionError("unexpected raw-data indices in figure source data")
    with h5py.File(source, "r") as handle:
        expected_response = np.asarray(
            handle[metadata["source_matlab_variable"]][::200, :]
        ).T
    if not np.array_equal(response_db, expected_response):
        raise AssertionError("figure source data do not match the committed raw array")

    figure_2c_mapping = figure_manifest["manuscript_figure_mapping"][0]
    expected_figure_2c_selection = (
        "response_db row 7 (output_port_id 7), all wavelengths"
    )
    if figure_2c_mapping["selection"] != expected_figure_2c_selection:
        raise AssertionError("Fig. 2c must map to output port 7")
    port7_peak_to_peak_db = float(np.ptp(response_db[6, :]))
    recorded_port7_peak_to_peak_db = figure_manifest["quality_checks"][
        "port7_peak_to_peak_db"
    ]
    if not np.isclose(
        port7_peak_to_peak_db,
        recorded_port7_peak_to_peak_db,
        rtol=0.0,
        atol=1e-12,
    ):
        raise AssertionError("Fig. 2c output-port-7 range does not match manifest")

    checked_files, parsed_manifests = verify_figure_package_checksums()
    checked_sample_files, sample_accuracy = verify_ovgg_sample()

    print(
        json.dumps(
            {
                "status": "ok",
                "source_sha256": metadata["source_sha256"],
                "source_variable": metadata["source_matlab_variable"],
                "source_shape": observed_shape,
                "figure_source_sha256": figure_manifest["derived_file"]["sha256"],
                "figure_response_shape": list(response_db.shape),
                "checksummed_figure_files": checked_files,
                "parsed_figure_manifests": parsed_manifests,
                "checksummed_ovgg_sample_files": checked_sample_files,
                "ovgg_sample_accuracy": sample_accuracy,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
