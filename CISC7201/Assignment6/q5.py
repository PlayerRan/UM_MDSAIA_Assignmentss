import geopandas as gpd
from shapely.geometry import Point, Polygon

def judge(lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4, lon5, lat5):
    # Create a polygon from the first four points
    polygon = Polygon([(lon1, lat1), (lon2, lat2), (lon3, lat3), (lon4, lat4)])
    
    # Create a point from the fifth point
    point = Point(lon5, lat5)
    
    # Check if the point is within the polygon
    if polygon.contains(point):
        return 'true'
    else:
        return 'false'

def input_coordinates():
    """
    Read coordinates for two points from standard input.
    """
    lon1, lat1 = map(float, input().strip().split())
    lon2, lat2 = map(float, input().strip().split())
    lon3, lat3 = map(float, input().strip().split())
    lon4, lat4 = map(float, input().strip().split())
    lon5, lat5 = map(float, input().strip().split())
    return lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4, lon5, lat5

lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4, lon5, lat5 = input_coordinates()
res = judge(lon1, lat1, lon2, lat2, lon3, lat3, lon4, lat4, lon5, lat5)
print(res)