"""Exact reference-region UNION for nearest-region distance, GEOS double precision.

These are uncertainty regions, never filled material surfaces. No convex hull
across different observations and no snapping/simplification tolerances.
"""
import numpy as np
import shapely


def convex_regions(rows):
    if not rows:return np.empty(0,dtype=object)
    vertices=np.array([r['vertices'] for r in rows])
    # Each single region is a derived convex trapezoid; this also supports
    # genuine zero-area line/point degeneracies without silently dropping it.
    return shapely.convex_hull(shapely.multipoints(vertices))


def prepare_reference(rows):
    groups={};result={}
    for row in rows:groups.setdefault((tuple(row['key']),row['source_axis']),[]).append(row)
    for token,rr in groups.items():result[token]=shapely.union_all(convex_regions(rr),grid_size=0)
    for key in {tuple(r['key']) for r in rows}:result[(key,None)]=shapely.union_all([geometry for (k,a),geometry in list(result.items()) if k==key and a is not None],grid_size=0)
    return result


def nearest_support_costs(current,reference,ray_count,cfg):
    cap=cfg['support_match_cap_m'];best=np.full(ray_count,cap);ordered=np.full(ray_count,cap);available=np.zeros(ray_count,bool);queries=0
    groups={}
    for r in current:groups.setdefault((tuple(r['key']),r['source_axis']),[]).append(r)
    for (key,a),rows in groups.items():
        ids=np.array([r['ray_id'] for r in rows]);available[ids]=True;geometry=convex_regions(rows)
        for source,values in [(None,best),(a,ordered)]:
            if (key,source) not in reference:continue
            distance=shapely.distance(geometry,reference[(key,source)]);assert np.isfinite(distance).all();np.minimum.at(values,ids,np.minimum(distance,cap));queries+=len(rows)
    def output(values):return {'mean_capped_squared_cost':float(np.mean((values/cap)**2)) if ray_count else 0.,'per_ray_distance_m':values,'hypothesis_available':available.copy(),'geometry_distance_queries':queries,'conditional_recurrence_prior':True,'free_birth_control_cost':0.}
    return output(best),output(ordered)
