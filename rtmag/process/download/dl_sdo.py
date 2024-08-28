from pathlib import Path
from sunpy.net import Fido, attrs as a
import astropy.units as u

def download_aia(download_path, start_time, end_time, wavelengths=[94, 171]):
    tr = a.Time(start_time, end_time)

    aia_results = {}
    aia_downloads = {}

    for w in wavelengths:
        print(f"Searching for AIA {w} data...")
        aia_results[str(w)] = Fido.search(
            tr,
            a.Instrument.aia, 
            a.Wavelength(w*u.angstrom))
        
    download_path = Path(download_path)
    for w, result in aia_results.items():
        print(f"Downloading AIA {w} data...")
        dl_path = download_path / str(w) / "{file}"
        aia_downloads[str(w)] = Fido.fetch(result, path=dl_path)

    for w in wavelengths:
        re_num = aia_results[str(w)].file_num
        dl_num = len(list((download_path / str(w)).glob("*.fits")))

        patience = 0
        while re_num != dl_num:
            print(aia_downloads[str(w)].errors)

            print(f"Error Downloading AIA {w} data..., retrying")
            Fido.fetch(aia_downloads[str(w)])

            re_num = aia_results[str(w)].file_num
            dl_num = len(list((download_path / str(w)).glob("*.fits")))
            patience += 1

            if patience > 10:
                print(f"Error Downloading AIA {w} data..., skipping")
                break


def download_sharp(download_path, start_time, end_time, harpnum, email):
    tr = a.Time(start_time, end_time)

    download_path = Path(download_path)

    download_path = download_path / "{file}"

    hmi_results = Fido.search(
        tr,
        a.jsoc.Notify(email),
        a.jsoc.Series("hmi.sharp_cea_720s"),
        a.jsoc.PrimeKey('HARPNUM', harpnum),
        a.jsoc.Segment("Bp") & a.jsoc.Segment("Bt") & a.jsoc.Segment("Br"),
    )

    hmi_downloads = Fido.fetch(hmi_results, path=download_path)