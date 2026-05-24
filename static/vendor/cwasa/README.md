# Vendored CWASA Runtime

These files are a local copy of the CWASA runtime used by BridgeSign's iframe
avatar player.

Source URLs fetched on 2026-05-21:

- https://vhg.cmp.uea.ac.uk/tech/jas/vhg2026/cwa/allcsa.js
- https://vhg.cmp.uea.ac.uk/tech/jas/vhg2026/cwa/cwasa.css
- https://vhg.cmp.uea.ac.uk/tech/jas/vhg2026/cwa/h2s.xsl
- https://vhg.cmp.uea.ac.uk/tech/jas/vhg2026/cwa/shaders/qskin.vert
- https://vhg.cmp.uea.ac.uk/tech/jas/vhg2026/cwa/shaders/qskin.frag
- https://vhg.cmp.uea.ac.uk/tech/jas/vhg2026/avatars/COMMON.jar
- https://vhg.cmp.uea.ac.uk/tech/jas/vhg2026/avatars/anna.jar

The app points `jasBase` at `/static/vendor/cwasa/vhg2026/` and uses only the
`anna` avatar, so CWASA does not need the remote UEA host at runtime for the
current BridgeSign avatar path.
