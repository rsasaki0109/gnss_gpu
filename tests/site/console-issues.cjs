// Only already-optional cosmetic resources are ignored. Resource errors do
// not necessarily include their URL in message.text(); Chrome supplies it in
// message.location(). Never ignore an app error based on matching its text.
function isOptionalAssetError(resourceURL,pageURL){
  try{
    const resource=new URL(resourceURL,pageURL),page=new URL(pageURL);
    return ["fonts.googleapis.com","fonts.gstatic.com"].includes(resource.hostname)
      ||(resource.origin===page.origin&&resource.pathname==="/favicon.ico");
  }catch{return false;}
}
module.exports={isOptionalAssetError};
