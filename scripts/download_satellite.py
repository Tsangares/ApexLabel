#!/usr/bin/env python3
"""
ArcGIS Satellite Image Downloader

Downloads satellite imagery tiles from ArcGIS World Imagery service.
Free for non-commercial use.

Usage:
    python scripts/download_satellite.py --lat 34.0522 --lon -118.2437 --count 100
    python scripts/download_satellite.py --lat 34.0522 --lon -118.2437 --count 500 --zoom 19
"""

import requests
import math
import os
import time
import argparse
from pathlib import Path


def get_tile(lat: float, lon: float, zoom: int = 18, size: int = 640) -> bytes | None:
    """
    Download a single satellite tile from ArcGIS.

    Args:
        lat: Latitude (decimal degrees)
        lon: Longitude (decimal degrees)
        zoom: Zoom level (18 = ~0.6m/pixel, 19 = ~0.3m/pixel)
        size: Image size in pixels (width and height)

    Returns:
        Image bytes or None on failure
    """
    # Calculate bounding box
    meters_per_pixel = 156543.03392 * math.cos(math.radians(lat)) / (2 ** zoom)
    extent_meters = size * meters_per_pixel
    lat_extent = extent_meters / 111000
    lon_extent = extent_meters / (111000 * math.cos(math.radians(lat)))

    bbox = {
        'min_lat': lat - lat_extent / 2,
        'max_lat': lat + lat_extent / 2,
        'min_lon': lon - lon_extent / 2,
        'max_lon': lon + lon_extent / 2,
    }

    url = "https://services.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/export"
    params = {
        'bbox': f"{bbox['min_lon']},{bbox['min_lat']},{bbox['max_lon']},{bbox['max_lat']}",
        'bboxSR': '4326',
        'imageSR': '4326',
        'size': f"{size},{size}",
        'format': 'png',
        'f': 'image'
    }

    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        return response.content
    except Exception as e:
        print(f"  Error: {e}")
        return None


def spiral_download(
    center_lat: float,
    center_lon: float,
    count: int = 100,
    zoom: int = 18,
    size: int = 640,
    output_dir: str = "data/sample_images",
    delay: float = 0.5
) -> list[str]:
    """
    Download satellite tiles in a spiral pattern from center point.

    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        count: Number of images to download
        zoom: Zoom level
        size: Image size in pixels
        output_dir: Output directory
        delay: Delay between requests (be nice to the server)

    Returns:
        List of downloaded file paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Calculate step size (non-overlapping tiles)
    meters_per_pixel = 156543.03392 * math.cos(math.radians(center_lat)) / (2 ** zoom)
    tile_meters = size * meters_per_pixel
    lat_step = tile_meters / 111000
    lon_step = tile_meters / (111000 * math.cos(math.radians(center_lat)))

    downloaded = []
    n = 0

    # Center tile
    print(f"[{n+1}/{count}] Downloading center tile...")
    img = get_tile(center_lat, center_lon, zoom, size)
    if img:
        filename = output_path / f"tile_{n:04d}.png"
        filename.write_bytes(img)
        downloaded.append(str(filename))
        n += 1
    time.sleep(delay)

    # Spiral outward
    ring = 1
    while n < count:
        # Starting position for this ring
        x, y = -ring, -ring

        # Four sides of the ring
        moves = [
            (1, 0, ring * 2),   # Right
            (0, 1, ring * 2),   # Up
            (-1, 0, ring * 2),  # Left
            (0, -1, ring * 2),  # Down
        ]

        for dx, dy, steps in moves:
            for _ in range(steps):
                if n >= count:
                    break

                lat = center_lat + y * lat_step
                lon = center_lon + x * lon_step

                print(f"[{n+1}/{count}] lat={lat:.6f}, lon={lon:.6f}")
                img = get_tile(lat, lon, zoom, size)

                if img:
                    filename = output_path / f"tile_{n:04d}.png"
                    filename.write_bytes(img)
                    downloaded.append(str(filename))
                    n += 1

                x += dx
                y += dy
                time.sleep(delay)

            if n >= count:
                break

        ring += 1

    print(f"\nDownloaded {len(downloaded)} images to {output_dir}/")
    return downloaded


def main():
    parser = argparse.ArgumentParser(
        description="Download satellite imagery from ArcGIS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 100 tiles around Los Angeles
  python scripts/download_satellite.py --lat 34.0522 --lon -118.2437 --count 100

  # Higher resolution (zoom 19)
  python scripts/download_satellite.py --lat 34.0522 --lon -118.2437 --count 50 --zoom 19

  # Custom output directory
  python scripts/download_satellite.py --lat 40.7128 --lon -74.0060 -o data/nyc_tiles

Common locations:
  Los Angeles:  34.0522, -118.2437
  New York:     40.7128, -74.0060
  San Francisco: 37.7749, -122.4194
  Chicago:      41.8781, -87.6298
        """
    )
    parser.add_argument('--lat', type=float, required=True, help='Center latitude')
    parser.add_argument('--lon', type=float, required=True, help='Center longitude')
    parser.add_argument('--count', '-n', type=int, default=100, help='Number of tiles (default: 100)')
    parser.add_argument('--zoom', '-z', type=int, default=18, help='Zoom level 1-19 (default: 18)')
    parser.add_argument('--size', '-s', type=int, default=640, help='Tile size in pixels (default: 640)')
    parser.add_argument('--output', '-o', type=str, default='data/sample_images', help='Output directory')
    parser.add_argument('--delay', '-d', type=float, default=0.5, help='Delay between requests (default: 0.5s)')

    args = parser.parse_args()

    print(f"=== ArcGIS Satellite Downloader ===")
    print(f"Center: {args.lat}, {args.lon}")
    print(f"Count: {args.count} tiles")
    print(f"Zoom: {args.zoom}")
    print(f"Output: {args.output}/")
    print()

    spiral_download(
        center_lat=args.lat,
        center_lon=args.lon,
        count=args.count,
        zoom=args.zoom,
        size=args.size,
        output_dir=args.output,
        delay=args.delay
    )


if __name__ == "__main__":
    main()
