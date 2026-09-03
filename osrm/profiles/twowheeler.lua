-- osrm/profiles/twowheeler.lua
--
-- Spec Section 2.3. OSRM ships car, bicycle and foot; India's dominant mode is the
-- two-wheeler, which is none of them. This profile is car.lua with three changes:
--
--   1. every speed is scaled by a factor read from
--      config/params/accessibility.yaml -> modes.two_wheeler.speed_factor;
--   2. access is allowed on highway=service, highway=track (grade1-2) and motorcycle=yes;
--   3. access is denied on motorway where motorcycle=no.
--
-- ---------------------------------------------------------------------------------------
-- The Lua / YAML parameter problem, and how it is solved here
-- ---------------------------------------------------------------------------------------
-- osrm-extract runs inside a Docker container with only /data and /opt mounted. The Lua
-- interpreter it embeds has no YAML parser, and shipping one would put a third-party Lua
-- dependency inside the routing container for no good reason. Equally, CONTRACT.md rule 1
-- forbids typing the speeds into a source file.
--
-- So the parameters are GENERATED. `ufe.layers.routing.write_twowheeler_constants(params,
-- path)` reads config/params/accessibility.yaml through the ordinary Params loader and
-- emits `twowheeler_speeds.lua`, a pure-data Lua module stamped with the params hash. This
-- profile requires that module and fails loudly if it is absent, so a stale or missing
-- regeneration can never silently fall back to a hardcoded number.
--
--   python -c "from ufe.params import load_params; \
--     from ufe.layers.routing import write_twowheeler_constants as w; \
--     w(load_params('vizag'), 'osrm/profiles/twowheeler_speeds.lua')"
--
-- Regenerate whenever accessibility.yaml changes, then re-run osrm-extract. The generated
-- file records the params hash so a matrix can be traced back to the parameter version that
-- produced it (spec Section 21, "stale cache across param versions").
--
-- NOTE: modes.two_wheeler.speed_factor is NOT yet present in accessibility.yaml. The
-- generator raises MissingParameter naming that path rather than assuming the 0.85 printed
-- in the Section 2.3 prose. Add the leaf (or pass speed_factor= explicitly) before running
-- osrm-extract with this profile.

api_version = 4

local car = require('car')
local ok, tuning = pcall(require, 'twowheeler_speeds')
if not ok then
  error(
    'twowheeler.lua: twowheeler_speeds.lua is missing. It is generated from ' ..
    'config/params/accessibility.yaml by ufe.layers.routing.write_twowheeler_constants; ' ..
    'no speed may be hardcoded in this profile (CONTRACT.md rule 1). See the header of ' ..
    'this file for the exact command.'
  )
end

local factor = tuning.speed_factor
if type(factor) ~= 'number' or factor <= 0 then
  error('twowheeler.lua: twowheeler_speeds.speed_factor must be a positive number')
end

-- Road classes whose absolute km/h the generator resolved from accessibility.yaml. Where a
-- generated class-specific speed exists it wins; every other OSM highway class keeps
-- car.lua's relative structure, scaled by the global factor.
local class_speed = {
  motorway        = tuning.speeds_kmh.expressway,
  motorway_link   = tuning.speeds_kmh.expressway,
  trunk           = tuning.speeds_kmh.national_highway,
  trunk_link      = tuning.speeds_kmh.national_highway,
  primary         = tuning.speeds_kmh.arterial,
  primary_link    = tuning.speeds_kmh.arterial,
  secondary       = tuning.speeds_kmh.collector,
  secondary_link  = tuning.speeds_kmh.collector,
  tertiary        = tuning.speeds_kmh.collector,
  tertiary_link   = tuning.speeds_kmh.collector,
  unclassified    = tuning.speeds_kmh['local'],
  residential     = tuning.speeds_kmh['local'],
  living_street   = tuning.speeds_kmh['local'],
  service         = tuning.speeds_kmh['local'],
  road            = tuning.speeds_kmh['local'],
  track           = tuning.speeds_kmh['local'],
}

function setup()
  local profile = car.setup()

  profile.properties.weight_name = 'routability'

  -- (1) scale every inherited speed, then override the classes we resolved from YAML
  for key, value in pairs(profile.speeds) do
    if type(value) == 'number' then
      profile.speeds[key] = value * factor
    end
  end
  for key, value in pairs(class_speed) do
    if type(value) == 'number' then
      profile.speeds[key] = value
    end
  end
  if profile.service_speeds then
    for key, value in pairs(profile.service_speeds) do
      if type(value) == 'number' then
        profile.service_speeds[key] = value * factor
      end
    end
  end

  -- (2) access. A two-wheeler is a motor_vehicle, so car.lua's access tag hierarchy is
  -- inherited; `motorcycle` is inserted ahead of it so an explicit motorcycle tag wins.
  table.insert(profile.access_tags_hierarchy, 1, 'motorcycle')
  table.insert(profile.restrictions, 1, 'motorcycle')

  -- service ways and grade1-2 tracks are usable by a two-wheeler even where car.lua
  -- excludes or penalises them
  profile.service_tag_forbidden = {}
  profile.avoid = profile.avoid or {}
  for i = #profile.avoid, 1, -1 do
    if profile.avoid[i] == 'track' or profile.avoid[i] == 'service' then
      table.remove(profile.avoid, i)
    end
  end
  profile.usable_track_grades = { grade1 = true, grade2 = true }

  return profile
end

-- (3) deny motorway where motorcycle=no, and apply the track-grade rule. Everything else is
-- car.lua's own way handler.
function process_way(profile, way, result, relations)
  local highway = way:get_value_by_key('highway')
  local motorcycle = way:get_value_by_key('motorcycle')

  if motorcycle == 'no' or motorcycle == 'private' then
    return
  end
  if highway == 'motorway' or highway == 'motorway_link' then
    if motorcycle == 'no' then
      return
    end
  end
  if highway == 'track' then
    local grade = way:get_value_by_key('tracktype')
    if grade ~= nil and not profile.usable_track_grades[grade] then
      return
    end
  end
  if highway == 'service' and motorcycle == nil then
    -- car.lua drops some service ways outright; a two-wheeler may use them.
    local access = way:get_value_by_key('access')
    if access == 'private' or access == 'no' then
      return
    end
  end

  car.process_way(profile, way, result, relations)
end

function process_node(profile, node, result, relations)
  return car.process_node(profile, node, result, relations)
end

function process_turn(profile, turn)
  return car.process_turn(profile, turn)
end

return {
  setup = setup,
  process_way = process_way,
  process_node = process_node,
  process_turn = process_turn,
}
