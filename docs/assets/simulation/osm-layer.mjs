import { groundFootprint, visibleTiles, tileMesh, lonLatToLocal, localToLonLat } from "./map-geometry.mjs";
import { VisibleTileCache } from "./tile-cache.mjs";

export const OSM_TILE_URL = tile => `https://tile.openstreetmap.org/${tile.z}/${tile.x}/${tile.y}.png`;

export class OSMBasemap {
  constructor(gl, scene, onChange, tileURL = OSM_TILE_URL) {
    this.gl = gl; this.scene = scene;
    this.height = (scene.provenance.geoid.min_m + scene.provenance.geoid.max_m) / 2;
    const [lon, lat] = localToLonLat([0,0,0], scene);
    this.planeZ = lonLatToLocal(lon, lat, this.height, scene)[2];
    const compile = (type, source) => {
      const shader = gl.createShader(type); gl.shaderSource(shader, source); gl.compileShader(shader);
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader));
      return shader;
    };
    const vertex = compile(gl.VERTEX_SHADER, `#version 300 es
      in vec3 a_position; in vec2 a_uv;
      uniform mat3 u_view; uniform vec3 u_target; uniform vec3 u_scale;
      out vec2 v_uv;
      void main() {
        vec3 p = u_view * (a_position - u_target);
        gl_Position = vec4(p * u_scale + vec3(0., -.12, 0.), 1.);
        v_uv = a_uv;
      }`);
    const fragment = compile(gl.FRAGMENT_SHADER, `#version 300 es
      precision highp float; uniform sampler2D u_tile;
      in vec2 v_uv; out vec4 color;
      void main() { color = vec4(mix(vec3(.055,.10,.135), texture(u_tile,v_uv).rgb, .85), 1.); }`);
    this.program = gl.createProgram(); gl.attachShader(this.program, vertex); gl.attachShader(this.program, fragment); gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw new Error(gl.getProgramInfoLog(this.program));
    gl.deleteShader(vertex); gl.deleteShader(fragment);
    this.uniforms = Object.fromEntries(["view", "target", "scale", "tile"].map(key => [key, gl.getUniformLocation(this.program, `u_${key}`)]));
    this.position = gl.getAttribLocation(this.program, "a_position"); this.uv = gl.getAttribLocation(this.program, "a_uv");
    this.cache = new VisibleTileCache(tile => this.load(tile, tileURL(tile)), onChange);
    this.lastPlan = "";
  }

  load(tile, url) {
    return new Promise((resolve, reject) => {
      const image = new Image();
      image.crossOrigin = "anonymous";
      image.referrerPolicy = "strict-origin-when-cross-origin";
      image.decoding = "async";
      image.onerror = () => reject(new Error("OpenStreetMap tile unavailable"));
      image.onload = () => {
        const gl = this.gl, texture = gl.createTexture(), vao = gl.createVertexArray(), buffer = gl.createBuffer();
        const dispose = () => { gl.deleteTexture(texture); gl.deleteVertexArray(vao); gl.deleteBuffer(buffer); };
        try {
          gl.bindTexture(gl.TEXTURE_2D, texture); gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false);
          gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
          gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR_MIPMAP_LINEAR); gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
          gl.generateMipmap(gl.TEXTURE_2D);
          const data = tileMesh(tile, this.scene, this.height);
          gl.bindVertexArray(vao); gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
          gl.enableVertexAttribArray(this.position); gl.vertexAttribPointer(this.position, 3, gl.FLOAT, false, 20, 0);
          gl.enableVertexAttribArray(this.uv); gl.vertexAttribPointer(this.uv, 2, gl.FLOAT, false, 20, 12);
          resolve({ texture, vao, count: data.length / 5, dispose });
        } catch (error) { dispose(); reject(error); }
      };
      // Default browser HTTP caching, UA and Referer; no prefetch/offline download.
      image.src = url;
    });
  }

  setEnabled(enabled) { this.cache.setEnabled(enabled); }

  draw(camera, width, height, scale, matrix) {
    const footprint = groundFootprint(camera, width, height, scale, this.planeZ);
    const tiles = visibleTiles(footprint, this.scene, scale), key = tiles.map(tile => tile.key).join(",");
    if (key !== this.lastPlan) { this.lastPlan = key; this.cache.select(tiles); }
    const ready = this.cache.ready();
    if (!ready.length) return false;
    const gl = this.gl;
    gl.useProgram(this.program); gl.disable(gl.DEPTH_TEST); gl.depthMask(false);
    gl.uniformMatrix3fv(this.uniforms.view, false, matrix); gl.uniform3fv(this.uniforms.target, camera.target);
    gl.uniform3f(this.uniforms.scale, 2 * scale / width, 2 * scale / height, 1 / 25000);
    gl.activeTexture(gl.TEXTURE0); gl.uniform1i(this.uniforms.tile, 0);
    for (const tile of ready) { gl.bindTexture(gl.TEXTURE_2D, tile.texture); gl.bindVertexArray(tile.vao); gl.drawArrays(gl.TRIANGLES, 0, tile.count); }
    gl.depthMask(true);
    return true;
  }
}
