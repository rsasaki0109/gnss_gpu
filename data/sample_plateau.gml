<?xml version="1.0" encoding="UTF-8"?>
<core:CityModel
  xmlns:core="http://www.opengis.net/citygml/2.0"
  xmlns:bldg="http://www.opengis.net/citygml/building/2.0"
  xmlns:gml="http://www.opengis.net/gml"
  xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
  xsi:schemaLocation="http://www.opengis.net/citygml/2.0 http://schemas.opengis.net/citygml/2.0/cityGMLBase.xsd
                       http://www.opengis.net/citygml/building/2.0 http://schemas.opengis.net/citygml/building/2.0/building.xsd">

  <!-- Building 1: Small box near Tokyo Station area -->
  <!-- Plane rectangular zone 9 coordinates (Y=northing, X=easting, Z=height) -->
  <!-- Approximate location: lat ~35.6812, lon ~139.7671 (Tokyo Station) -->
  <core:cityObjectMember>
    <bldg:Building gml:id="bldg_001">
      <bldg:measuredHeight uom="m">20.0</bldg:measuredHeight>
      <bldg:lod1Solid>
        <gml:Solid>
          <gml:exterior>
            <gml:CompositeSurface>
              <!-- Bottom face -->
              <gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>
                        -35305.0 -2845.0 5.0
                        -35305.0 -2825.0 5.0
                        -35285.0 -2825.0 5.0
                        -35285.0 -2845.0 5.0
                        -35305.0 -2845.0 5.0
                      </gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
              <!-- Top face -->
              <gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>
                        -35305.0 -2845.0 25.0
                        -35285.0 -2845.0 25.0
                        -35285.0 -2825.0 25.0
                        -35305.0 -2825.0 25.0
                        -35305.0 -2845.0 25.0
                      </gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
              <!-- Front face (south) -->
              <gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>
                        -35305.0 -2845.0 5.0
                        -35285.0 -2845.0 5.0
                        -35285.0 -2845.0 25.0
                        -35305.0 -2845.0 25.0
                        -35305.0 -2845.0 5.0
                      </gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
              <!-- Back face (north) -->
              <gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>
                        -35305.0 -2825.0 5.0
                        -35305.0 -2825.0 25.0
                        -35285.0 -2825.0 25.0
                        -35285.0 -2825.0 5.0
                        -35305.0 -2825.0 5.0
                      </gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
              <!-- Left face (west) -->
              <gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>
                        -35305.0 -2845.0 5.0
                        -35305.0 -2845.0 25.0
                        -35305.0 -2825.0 25.0
                        -35305.0 -2825.0 5.0
                        -35305.0 -2845.0 5.0
                      </gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
              <!-- Right face (east) -->
              <gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>
                        -35285.0 -2845.0 5.0
                        -35285.0 -2825.0 5.0
                        -35285.0 -2825.0 25.0
                        -35285.0 -2845.0 25.0
                        -35285.0 -2845.0 5.0
                      </gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
            </gml:CompositeSurface>
          </gml:exterior>
        </gml:Solid>
      </bldg:lod1Solid>
    </bldg:Building>
  </core:cityObjectMember>

  <!-- Building 2: Taller building slightly to the east -->
  <core:cityObjectMember>
    <bldg:Building gml:id="bldg_002">
      <bldg:measuredHeight uom="m">45.0</bldg:measuredHeight>
      <bldg:lod1Solid>
        <gml:Solid>
          <gml:exterior>
            <gml:CompositeSurface>
              <!-- Bottom face -->
              <gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>
                        -35310.0 -2800.0 5.0
                        -35310.0 -2770.0 5.0
                        -35280.0 -2770.0 5.0
                        -35280.0 -2800.0 5.0
                        -35310.0 -2800.0 5.0
                      </gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
              <!-- Top face -->
              <gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>
                        -35310.0 -2800.0 50.0
                        -35280.0 -2800.0 50.0
                        -35280.0 -2770.0 50.0
                        -35310.0 -2770.0 50.0
                        -35310.0 -2800.0 50.0
                      </gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
              <!-- Front face -->
              <gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>
                        -35310.0 -2800.0 5.0
                        -35280.0 -2800.0 5.0
                        -35280.0 -2800.0 50.0
                        -35310.0 -2800.0 50.0
                        -35310.0 -2800.0 5.0
                      </gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
              <!-- Back face -->
              <gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>
                        -35310.0 -2770.0 5.0
                        -35310.0 -2770.0 50.0
                        -35280.0 -2770.0 50.0
                        -35280.0 -2770.0 5.0
                        -35310.0 -2770.0 5.0
                      </gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
              <!-- Left face -->
              <gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>
                        -35310.0 -2800.0 5.0
                        -35310.0 -2800.0 50.0
                        -35310.0 -2770.0 50.0
                        -35310.0 -2770.0 5.0
                        -35310.0 -2800.0 5.0
                      </gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
              <!-- Right face -->
              <gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>
                        -35280.0 -2800.0 5.0
                        -35280.0 -2770.0 5.0
                        -35280.0 -2770.0 50.0
                        -35280.0 -2800.0 50.0
                        -35280.0 -2800.0 5.0
                      </gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
            </gml:CompositeSurface>
          </gml:exterior>
        </gml:Solid>
      </bldg:lod1Solid>
    </bldg:Building>
  </core:cityObjectMember>

  <!-- Building 3: Small L-shaped footprint (simplified as rectangle) -->
  <core:cityObjectMember>
    <bldg:Building gml:id="bldg_003">
      <bldg:measuredHeight uom="m">12.0</bldg:measuredHeight>
      <bldg:lod1Solid>
        <gml:Solid>
          <gml:exterior>
            <gml:CompositeSurface>
              <!-- Bottom face -->
              <gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>
                        -35270.0 -2845.0 5.0
                        -35270.0 -2830.0 5.0
                        -35255.0 -2830.0 5.0
                        -35255.0 -2845.0 5.0
                        -35270.0 -2845.0 5.0
                      </gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
              <!-- Top face -->
              <gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>
                        -35270.0 -2845.0 17.0
                        -35255.0 -2845.0 17.0
                        -35255.0 -2830.0 17.0
                        -35270.0 -2830.0 17.0
                        -35270.0 -2845.0 17.0
                      </gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
              <!-- Front face -->
              <gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>
                        -35270.0 -2845.0 5.0
                        -35255.0 -2845.0 5.0
                        -35255.0 -2845.0 17.0
                        -35270.0 -2845.0 17.0
                        -35270.0 -2845.0 5.0
                      </gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
              <!-- Back face -->
              <gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>
                        -35270.0 -2830.0 5.0
                        -35270.0 -2830.0 17.0
                        -35255.0 -2830.0 17.0
                        -35255.0 -2830.0 5.0
                        -35270.0 -2830.0 5.0
                      </gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
              <!-- Left face -->
              <gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>
                        -35270.0 -2845.0 5.0
                        -35270.0 -2845.0 17.0
                        -35270.0 -2830.0 17.0
                        -35270.0 -2830.0 5.0
                        -35270.0 -2845.0 5.0
                      </gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
              <!-- Right face -->
              <gml:surfaceMember>
                <gml:Polygon>
                  <gml:exterior>
                    <gml:LinearRing>
                      <gml:posList>
                        -35255.0 -2845.0 5.0
                        -35255.0 -2830.0 5.0
                        -35255.0 -2830.0 17.0
                        -35255.0 -2845.0 17.0
                        -35255.0 -2845.0 5.0
                      </gml:posList>
                    </gml:LinearRing>
                  </gml:exterior>
                </gml:Polygon>
              </gml:surfaceMember>
            </gml:CompositeSurface>
          </gml:exterior>
        </gml:Solid>
      </bldg:lod1Solid>
    </bldg:Building>
  </core:cityObjectMember>

</core:CityModel>
