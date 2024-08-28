import drms
from astropy.time import Time

def find_ar(d, show_image=False):

    st = Time(d)
    c = drms.Client()
    ar = c.query("hmi.Mharp_720s[][%sZ]" % st,
                    key = ["NOAA_AR", "HARPNUM"])

    if show_image:
        yr = st.iso[:4]
        mo = st.iso[5:7]
        da = st.iso[8:10]
        hr = st.iso[11:13]
        return ar, f"http://jsoc.stanford.edu/doc/data/hmi/harp/harp_definitive/{yr}/{mo}/{da}/harp.{yr}.{mo}.{da}_{hr}:00:00_TAI.png"
    
    return ar, None