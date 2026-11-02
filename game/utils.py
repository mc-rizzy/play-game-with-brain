def lerp(a, b, t): return a + (b - a) * t
def lerp_col(c1, c2, t): return tuple(int(lerp(c1[i], c2[i], t)) for i in range(3))

def sky_and_daylight(tod):
    """tod 0..1 (0.25=sunrise, 0.5=noon, 0.75=sunset) -> (sky_top, sky_bot, daylight)"""
    keys = [(0.00,(16,18,48),(30,34,70),.10),(0.22,(70,80,130),(140,120,120),.25),
            (0.30,(110,170,230),(180,225,245),.95),(0.50,(100,175,235),(190,230,250),1.0),
            (0.70,(110,165,225),(240,190,140),.9),(0.78,(240,140,90),(250,180,120),.35),
            (0.85,(24,26,64),(60,50,90),.10),(1.00,(16,18,48),(30,34,70),.10)]
    for i in range(len(keys) - 1):
        t0, top0, bot0, d0 = keys[i]; t1, top1, bot1, d1 = keys[i + 1]
        if t0 <= tod <= t1:
            f = (tod - t0) / max(1e-6, t1 - t0)
            return lerp_col(top0, top1, f), lerp_col(bot0, bot1, f), lerp(d0, d1, f)
    return keys[0][1], keys[0][2], keys[0][3]