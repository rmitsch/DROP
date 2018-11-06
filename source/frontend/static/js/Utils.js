export default class Utils
{
    /**
     * Generates a UUID(-like) object that can be used as quasi-unique ID for HTML elements.
     * @returns {string | * | void}
     */
    static uuidv4()
    {
        // Attach "id_" as prefix since CSS3 can't handle IDs starting with digits.
        return "id_" + ([1e7]+-1e3+-4e3+-8e3+-1e11).replace(/[018]/g, c =>
          (c ^ crypto.getRandomValues(new Uint8Array(1))[0] & 15 >> c / 4).toString(16)
        )
    }

    /**
     * Returns type of object.
     * Source: https://stackoverflow.com/questions/7390426/better-way-to-get-type-of-a-javascript-variable
     * @param obj
     * @returns {string}
     */
    static toType(obj)
    {
      return ({}).toString.call(obj).match(/\s([a-zA-Z]+)/)[1].toLowerCase()
    }

    /**
     * Spawns new child in specified parent.
     * @param parentDivID
     * @param childDivID
     * @param childDivCSSClasses
     * @param text
     * @returns {HTMLDivElement}
     */
    static spawnChildDiv(parentDivID, childDivID, childDivCSSClasses, text)
    {
        let div         = document.createElement('div');
        // If no child div ID specified: Generate random ID.
        div.id         = (typeof childDivID == "undefined") || (childDivID == null) ? Utils.uuidv4() : childDivID;
        div.className  = childDivCSSClasses;
        if (text != null && typeof text != "undefined")
            div.innerHTML = text;
        $("#" + parentDivID).append(div);

        return div;
    }

    /**
     * Spawns new child in specified parent.
     * @param parentDivID
     * @param childSpanID
     * @param childSpanCSSClasses
     * @param text
     * @returns {HTMLSpanElement}
     */
    static spawnChildSpan(parentDivID, childSpanID, childSpanCSSClasses, text)
    {

        let span        = document.createElement('span');
        span.id         = (typeof childSpanID == "undefined") || (childSpanID == null) ? Utils.uuidv4() : childSpanID;
        span.className  = childSpanCSSClasses;
        if (text != null && typeof text != "undefined")
            span.innerHTML = text;
        $("#" + parentDivID).append(span);

        return span;
    }

    /**
     * Unfolds list of hyperparamter objects in list of hyperparameter names.
     * @param hyperparameterObjectList
     * @returns {Array}
     */
    static unfoldHyperparameterObjectList(hyperparameterObjectList)
    {
        let hyperparameterNames = [];
        for (let hyperparam in hyperparameterObjectList) {
            hyperparameterNames.push(hyperparameterObjectList[hyperparam].name);
        }

        return hyperparameterNames;
    }

    /**
     * Remove empty bins. Extended by functionality to add top() and bottom().
     * https://github.com/dc-js/dc.js/wiki/FAQ#remove-empty-bins
     * @param group
     * @returns {{all: all, top: top, bottom: bottom}}
     */
    static removeEmptyBins(group)
    {
        return {
            all: function () {
                return group.all().filter(function(d) {
                    return d.value !== 0;
                });
            },

            top: function(N) {
                return group.top(N).filter(function(d) {
                    return d.value !== 0;
                });
            },

            bottom: function(N) {
                return group.top(Infinity).slice(-N).reverse().filter(function(d) {
                    return d.value !== 0;
                });
            }
        };
    }

    /**
     * Debounces a callback listener.
     * Source: https://remysharp.com/2010/07/21/throttling-function-calls
     * @param fn
     * @param delay
     * @returns {Function}
     */
    static debounce(fn, delay)
    {
        var timer = null;
        return function () {
            var context = this, args = arguments;
            clearTimeout(timer);
            timer = setTimeout(function () {
                fn.apply(context, args);
            }, delay);
        };
    }

    /**
     * Parse GET parameter string for specific parameter.
     * Source: https://stackoverflow.com/questions/5448545/how-to-retrieve-get-parameters-from-javascript
     * @param parameterName
     * @returns Parsed result.
     */
    static findGETParameter(parameterName)
    {
        let result  = null;
        let tmp     = [];

        location.search
            .substr(1)
            .split("&")
            .forEach(function (item) {
                tmp = item.split("=");
                if (tmp[0] === parameterName) result = decodeURIComponent(tmp[1]);
            });

        return result;
    }

    /**
     * Round value to nearest step.
     * @param value
     * @param step
     * @returns {number}
     */
    static round(value, step)
    {
        step || (step = 1.0);
        let inv = 1.0 / step;
        return Math.round(value * inv) / inv;
    }

    /**
     * Floors value to nearest step.
     * @param value
     * @param step
     * @returns {number}
     */
    static floor(value, step)
    {
        step || (step = 1.0);
        let inv = 1.0 / step;
        return Math.floor(value * inv) / inv;
    }

    /**
     * Ceils value to nearest step.
     * @param value
     * @param step
     * @returns {number}
     */
    static ceil(value, step)
    {
        step || (step = 1.0);
        let inv = 1.0 / step;
        return Math.ceil(value * inv) / inv;
    }

    /**
     * Binary search in sorted list. Source:
     * https://stackoverflow.com/questions/10264239/fastest-way-to-determine-if-an-element-is-in-a-sorted-array.
     * @param records
     * @param attribute
     * @param value
     * @returns {number}
     */
    static binarySearch(records, attribute, value)
    {
        let startIndex = 0,
            stopIndex = records.length - 1,
            middle = Math.floor((stopIndex + startIndex) / 2);

        while (records[middle][attribute] !== value && startIndex < stopIndex) {
            //adjust search area
            if (value < records[middle][attribute])
                stopIndex = middle - 1;
            else if (value > records[middle][attribute])
                startIndex = middle + 1;

            //recalculate middle
            middle = Math.floor((stopIndex + startIndex) / 2);
        }

        //make sure it's the right value
        return (records[middle][attribute] !== value) ? -1 : middle;
    }
}
