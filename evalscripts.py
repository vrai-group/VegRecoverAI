evalscript_raw = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B04",
                        "B03",
                        "B08",
                        "dataMask",
                        "CLM"],
                units: "DN"
            }],
            output: {
                bands: 2,
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(sample) {
        if (sample.dataMask == 1)  {
            if (sample.CLM == 0) {
                let NDVI = (sample.B08 - sample.B04) / (sample.B08 + sample.B04)
                let GNDVI = (sample.B08 - sample.B03) / (sample.B08 + sample.B03)
                return [NDVI, GNDVI]
            } else {
                return [NaN, NaN]
            }
        } else {
            return [NaN, NaN]
        }
    }
"""
evalscript_raw_snow = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B04",
                        "B03",
                        "B08",
                        "dataMask",
                        "CLM",
                        "SNW"],
                units: "DN"
            }],
            output: {
                bands: 2,
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(sample) {
        if (sample.dataMask == 1 || sample.SNW < 0.5)  {
            if (sample.CLM == 0) {
                let NDVI = (sample.B08 - sample.B04) / (sample.B08 + sample.B04)
                let GNDVI = (sample.B08 - sample.B03) / (sample.B08 + sample.B03)
                return [NDVI, GNDVI]
            } else {
                return [NaN, NaN]
            }
        } else {
            return [NaN, NaN]
        }
    }
"""

evalscript_raw_landsat8 = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B04",
                        "B05",
                        "dataMask"],
                units: "REFLECTANCE"
            }],
            output: {
                bands: 1,
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(sample) {
        if (sample.dataMask == 1)  {
            let NDVI = (sample.B05 - sample.B04) / (sample.B05 + sample.B04)
            return [NDVI]
        } else {
            return [NaN]
        }
    }
"""

evalscript_raw_landsat7 = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B04",
                        "B03",
                        "dataMask"],
                units: "REFLECTANCE"
            }],
            output: {
                bands: 1,
                sampleType: "FLOAT32"
            }
        };
    }

    function evaluatePixel(sample) {
        if (sample.dataMask == 1)  {
            let NDVI = (sample.B04 - sample.B03) / (sample.B04 + sample.B03)
            return [NDVI]
        } else {
            return [NaN]
        }
    }
"""


evalscript = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B07"]
            }],
            output: {
                bands: 1
            }
        };
    }

    function evaluatePixel(sample) {
        return [sample.B07, sample.B07, sample.B07];
    }
"""


evalscript_true_color = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B02", "B03", "B04"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [3.5*sample.B04, 3.5*sample.B03, 3.5*sample.B02];
    }
"""

evalscript_true_color_ls7 = """
    //VERSION=3

    function setup() {
        return {
            input: [{
                bands: ["B01", "B02", "B03"]
            }],
            output: {
                bands: 3
            }
        };
    }

    function evaluatePixel(sample) {
        return [3*sample.B03, 3.5*sample.B02, 3.5*sample.B01];
    }
"""