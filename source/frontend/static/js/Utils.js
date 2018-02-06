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
     * Spawns new child inspecified parent.
     * @param parentDivID
     * @param childDivID
     * @param childDivCSSClasses
     * @param text
     * @returns {HTMLDivElement}
     */
    static spawnChildDiv(parentDivID, childDivID, childDivCSSClasses, text)
    {
        let div = document.createElement('div');
        div.id         = (typeof childDivID == "undefined") || (childDivID == null) ? Utils.uuidv4() : childDivID;
        div.className  = childDivCSSClasses;
        if (text != null && typeof text != "undefined")
            div.innerHTML  = text;
        $("#" + parentDivID).append(div);

        return div;
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
    static remove_empty_bins(group)
    {
        return {
            all: function () {
                return group.all().filter(function(d) {
                    return d.value.count !== 0;
                });
            },

            top: function(N) {
                return group.top(N).filter(function(d) {
                    return d.value.count !== 0;
                });
            },

            bottom: function(N) {
                return group.top(Infinity).slice(-N).reverse().filter(function(d) {
                    return d.value.count !== 0;
                });
            }
        };
    }

}
