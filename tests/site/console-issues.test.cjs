const {test}=require("node:test");
const assert=require("node:assert/strict");
const {isOptionalAssetError}=require("./console-issues.cjs");
const page="http://127.0.0.1:4173/gnss_gpu/results.html";
test("optional font resources are identified by their actual location",()=>{
  assert.equal(isOptionalAssetError("https://fonts.googleapis.com/css2?family=Test",page),true);
  assert.equal(isOptionalAssetError("https://fonts.gstatic.com/test.woff2",page),true);
});
test("only the same-origin root favicon is optional",()=>{
  assert.equal(isOptionalAssetError("http://127.0.0.1:4173/favicon.ico",page),true);
  assert.equal(isOptionalAssetError("https://example.com/favicon.ico",page),false);
});
test("application and other resource failures remain errors",()=>{
  for(const url of ["assets/results_snapshot.js","assets/missing.png","site.css?message=fonts.googleapis.com","https://fonts.googleapis.com.evil.test/file",""])assert.equal(isOptionalAssetError(url,page),false);
});
