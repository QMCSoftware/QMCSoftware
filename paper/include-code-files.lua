-- include-code-files.lua
-- Inline inclusion of external files for Pandoc.
-- Supports two syntaxes:
-- 1) Raw line:    !INCLUDE "path/to/file.ext"   (auto-wrapped in fenced code by extension)
-- 2) Fenced code: ```{.lang include="path/to/file"} ... ``` (content replaced, lang preserved)

local function read_file(path)
  local fh = io.open(path, "r")
  if not fh then
    io.stderr:write("[include-code-files] Could not open file: " .. path .. "\n")
    return nil
  end
  local content = fh:read("*a")
  fh:close()
  return content
end

-- 1) Handle raw include directive lines: !INCLUDE "..."
function RawBlock(el)
  -- Only process plain raw blocks in markdown
  if el.format ~= "markdown" and el.format ~= "" then return nil end
  local filename = el.text:match('^!INCLUDE%s+"([^"]+)"%s*$')
  if not filename then return nil end

  local content = read_file(filename)
  if not content then
    -- Drop the line to avoid raw literal leak
    return {}
  end

  -- Infer language from extension if any
  local lang = filename:match("%.([^.]+)$") or ""
  -- Build a code block as markdown then read into Pandoc AST
  local fence = string.format("```%s\n%s\n```", lang, content)
  return pandoc.read(fence, "markdown").blocks
end

-- 2) Handle fenced code with include attribute
function CodeBlock(cb)
  -- cb.attr is a pandoc.Attr: identifier, classes, attributes (keyvals)
  local id = cb.attr.identifier
  local classes = cb.attr.classes
  local kv = cb.attr.attributes or {}

  local inc = kv["include"] or kv["src"]
  if not inc or inc == "" then
    return nil
  end

  local content = read_file(inc)
  if not content then
    -- If file missing, keep original block (so build surfaces it)
    return nil
  end

  -- Optional line range slicing: support start-line/end-line and start_line/end_line
  local function to_int(x)
    if not x then return nil end
    local s = tostring(x)
    local n = tonumber(s:match("^%s*(-?%d+)%s*$"))
    return n
  end

  local start_line = to_int(kv["start-line"]) or to_int(kv["start_line"]) or nil
  local end_line   = to_int(kv["end-line"])   or to_int(kv["end_line"])   or nil

  if start_line or end_line then
    -- Split into 1-based line array
    local lines = {}
    for line in (content .. "\n"):gmatch("(.-)\n") do
      table.insert(lines, line)
    end
    local n = #lines
    local s = start_line or 1
    local e = end_line or n
    if s < 1 then s = 1 end
    if e > n then e = n end
    if s <= e and n > 0 then
      local slice = {}
      for i = s, e do
        table.insert(slice, lines[i])
      end
      content = table.concat(slice, "\n") .. "\n"
    else
      content = "\n"
    end
  end

  -- Preserve any existing language class (e.g., { .python include=... })
  -- If no class, try infer from extension for better highlighting
  if #classes == 0 then
    local ext = inc:match("%.([^.]+)$")
    if ext and ext ~= "" then
      classes = { ext }
    end
  end

  -- Return a new CodeBlock with the included content
  local new_attr = pandoc.Attr(id or "", classes, {})
  return pandoc.CodeBlock(content, new_attr)
end

return {
  { RawBlock = RawBlock, CodeBlock = CodeBlock }
}
