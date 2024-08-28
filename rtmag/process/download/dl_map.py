import drms
import numpy as np
from astropy.time import Time
from astropy.io import fits
from sunpy.map import Map

def get_sharp_map(d, harpnum):
    st = Time(d)

    yr = st.iso[:4]
    mo = st.iso[5:7]
    da = st.iso[8:10]
    hr = st.iso[11:13]
    mi = st.iso[14:16]

    c = drms.Client()

    hmi_query = f'hmi.sharp_cea_720s[{harpnum}][{yr}.{mo}.{da}_{hr}:{mi}:00_TAI]'
    hmi_keys, hmi_segments = c.query(hmi_query, key=drms.const.all, seg='Bp, Bt, Br, magnetogram')
    print("T_REC: ", hmi_keys.T_REC[0])

    hmi_Bp_url = 'http://jsoc.stanford.edu' + hmi_segments.Bp[0]
    hmi_Bt_url = 'http://jsoc.stanford.edu' + hmi_segments.Bt[0]
    hmi_Br_url = 'http://jsoc.stanford.edu' + hmi_segments.Br[0]

    hmi_Br = fits.open(hmi_Br_url)
    hmi_Bt = fits.open(hmi_Bt_url)
    hmi_Bp = fits.open(hmi_Bp_url)

    mag_url = 'http://jsoc.stanford.edu' + hmi_segments.magnetogram[0]
    mag_image = fits.open(mag_url)
    mag_header = dict(hmi_keys.iloc[0])
    hmi_map = Map(mag_image[1].data, mag_header)

    # Bx = Bp, By = -Bt, Bz = Br
    hmi_data = np.stack([hmi_Bp[1].data, -hmi_Bt[1].data, hmi_Br[1].data]).T
    hmi_data = np.nan_to_num(hmi_data, nan=0.0)
    hmi_data = hmi_data.astype(np.float32)
    hmi_data.shape

    return hmi_map, hmi_data


def get_aia_map(d, wavelength=171):
    st = Time(d)

    yr = st.iso[:4]
    mo = st.iso[5:7]
    da = st.iso[8:10]
    hr = st.iso[11:13]
    mi = st.iso[14:16]

    c = drms.Client()

    wavelength = str(wavelength)
    
    aia_query = f'aia.lev1_euv_12s[{yr}-{mo}-{da}T{hr}:00:00Z][{wavelength}]'
    aia_keys, aia_segments = c.query(aia_query, key=drms.const.all, seg='image')
    print("T_REC: ", aia_keys.T_REC[0])

    aia_url = 'http://jsoc.stanford.edu' + aia_segments.image[0]
    aia_image = fits.open(aia_url)
    aia_header = dict(aia_keys.iloc[0])
    aia_image.verify('fix')
    aia_map = Map(aia_image[1].data, aia_header)

    return aia_map