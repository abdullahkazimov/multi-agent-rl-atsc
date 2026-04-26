"""
generate_seeds.py
=================
Utility to generate seed-specific SUMO route files for evaluation.

Each seed-specific route file uses the same traffic demands as the base
route file but with Poisson-distributed departure times re-sampled from
a seed-specific random number generator.  This creates independent demand
realisations (thesis §3.7 — 'seed-specific SUMO route files').

Usage
-----
  python generate_seeds.py --scenario main
  python generate_seeds.py --scenario bottleneck --seeds 42 123 456 789 1000
"""

from __future__ import annotations
import os
import argparse
import xml.etree.ElementTree as ET
from typing import List, Dict, Tuple
import numpy as np

from envs.scenario_configs import get_scenario_config, SCENARIOS


def parse_flows(rou_xml_path: str) -> Tuple[ET.Element, List[dict]]:
    """Parse <vType> and <flow> elements from a route XML file."""
    tree = ET.parse(rou_xml_path)
    root = tree.getroot()

    flows = []
    for flow in root.findall("flow"):
        attrib = flow.attrib.copy()
        flows.append(attrib)

    return root, flows


def poisson_departures(
    veh_per_hour: float,
    begin: float,
    end: float,
    rng: np.random.Generator,
) -> List[float]:
    """
    Sample Poisson-distributed departure times from a flow.
    Returns sorted list of departure times in [begin, end).
    """
    rate     = veh_per_hour / 3600.0
    duration = end - begin
    n_veh    = rng.poisson(rate * duration)
    if n_veh == 0:
        return []
    times = rng.uniform(begin, end, n_veh)
    return sorted(times.tolist())


def generate_seed_route_file(
    rou_xml_path: str,
    output_path:  str,
    seed:         int,
):
    """
    Create a seed-specific route file by converting <flow> elements to
    individual <vehicle> elements with Poisson-sampled departure times.
    """
    rng  = np.random.default_rng(seed)
    tree = ET.parse(rou_xml_path)
    root = tree.getroot()

    # Collect vTypes
    vtypes = {vt.attrib["id"]: vt for vt in root.findall("vType")}
    # Collect routes
    routes = {r.attrib["id"]: r for r in root.findall("route")}
    # Collect flows
    flows  = root.findall("flow")

    new_root = ET.Element("routes")
    new_root.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    new_root.set("xsi:noNamespaceSchemaLocation",
                 "http://sumo.dlr.de/xsd/routes_file.xsd")

    # Copy vTypes
    for vt in vtypes.values():
        new_root.append(vt)

    # Copy routes
    for rt in routes.values():
        new_root.append(rt)

    # Convert flows to vehicles
    veh_id = 0
    for flow in flows:
        a         = flow.attrib
        begin     = float(a.get("begin", 0))
        end       = float(a.get("end",   3600))
        vph       = float(a.get("vehsPerHour", a.get("number", 600)))
        vtype_id  = a.get("type", list(vtypes.keys())[0] if vtypes else "car")
        from_edge = a.get("from", "")
        to_edge   = a.get("to",   "")
        via_str   = a.get("via",  "")
        route_id  = a.get("route", "")

        dep_times = poisson_departures(vph, begin, end, rng)
        for t in dep_times:
            veh = ET.SubElement(new_root, "vehicle")
            veh.set("id",    f"v_{veh_id}")
            veh.set("type",  vtype_id)
            veh.set("depart", f"{t:.2f}")
            veh.set("departLane", a.get("departLane", "free"))
            veh.set("departSpeed", a.get("departSpeed", "random"))

            if route_id and route_id in routes:
                veh.set("route", route_id)
            elif from_edge and to_edge:
                trip = ET.SubElement(veh, "route")
                edges = f"{from_edge}"
                if via_str:
                    edges += f" {via_str}"
                edges += f" {to_edge}"
                trip.set("edges", edges)
            veh_id += 1

    # Sort vehicles by departure time
    vehicles = new_root.findall("vehicle")
    for v in vehicles:
        new_root.remove(v)
    vehicles.sort(key=lambda v: float(v.get("depart", 0)))
    for v in vehicles:
        new_root.append(v)

    ET.indent(new_root, space="    ")
    ET.ElementTree(new_root).write(output_path, encoding="unicode", xml_declaration=True)


def generate_scenario_seeds(
    scenario_name: str,
    seeds:         List[int] | None = None,
):
    cfg      = get_scenario_config(scenario_name)
    seeds    = seeds or cfg["eval_seeds"]
    base_rou = os.path.join(cfg["scenario_dir"], "baku.rou.xml")

    print(f"Generating seed-specific route files for '{scenario_name}'…")
    for seed in seeds:
        out_path = os.path.join(cfg["scenario_dir"], f"baku_seed_{seed}.rou.xml")
        if os.path.exists(out_path):
            print(f"  seed {seed:>5}  already exists — skipped")
            continue
        generate_seed_route_file(base_rou, out_path, seed)
        print(f"  seed {seed:>5}  → {out_path}")

    print("Done.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Generate seed-specific route files")
    p.add_argument("--scenario", required=True,
                   choices=list(SCENARIOS.keys()))
    p.add_argument("--seeds", nargs="+", type=int, default=None)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    generate_scenario_seeds(args.scenario, seeds=args.seeds)
