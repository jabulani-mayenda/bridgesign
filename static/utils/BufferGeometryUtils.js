import {
  TriangleFanDrawMode,
  TriangleStripDrawMode
} from 'three';

function getIndexValue(index, position) {
  return index ? index.getX(position) : position;
}

function toTrianglesDrawMode(geometry, drawMode) {
  if (drawMode !== TriangleFanDrawMode && drawMode !== TriangleStripDrawMode) {
    console.warn('THREE.BufferGeometryUtils.toTrianglesDrawMode(): Unsupported draw mode:', drawMode);
    return geometry;
  }

  const index = geometry.getIndex();
  const position = geometry.getAttribute('position');
  if (!position) return geometry;

  const sourceCount = index ? index.count : position.count;
  const triangleCount = sourceCount - 2;
  if (triangleCount <= 0) return geometry.clone();

  const indices = [];
  for (let i = 0; i < triangleCount; i += 1) {
    if (drawMode === TriangleFanDrawMode) {
      indices.push(
        getIndexValue(index, 0),
        getIndexValue(index, i + 1),
        getIndexValue(index, i + 2)
      );
    } else if (i % 2 === 0) {
      indices.push(
        getIndexValue(index, i),
        getIndexValue(index, i + 1),
        getIndexValue(index, i + 2)
      );
    } else {
      indices.push(
        getIndexValue(index, i + 2),
        getIndexValue(index, i + 1),
        getIndexValue(index, i)
      );
    }
  }

  const newGeometry = geometry.clone();
  newGeometry.setIndex(indices);
  return newGeometry;
}

export { toTrianglesDrawMode };
