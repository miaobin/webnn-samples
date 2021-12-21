/*!
 * ONNX Runtime Common v1.9.0
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */
!function(){"use strict";var e={d:function(r,t){for(var n in t)e.o(t,n)&&!e.o(r,n)&&Object.defineProperty(r,n,{enumerable:!0,get:t[n]})},o:function(e,r){return Object.prototype.hasOwnProperty.call(e,r)},r:function(e){"undefined"!=typeof Symbol&&Symbol.toStringTag&&Object.defineProperty(e,Symbol.toStringTag,{value:"Module"}),Object.defineProperty(e,"__esModule",{value:!0})}},r={};e.r(r),e.d(r,{InferenceSession:function(){return b},Tensor:function(){return c},env:function(){return a},registerBackend:function(){return o}});var t={},n=[],o=function(e,r,o){if(!r||"function"!=typeof r.init||"function"!=typeof r.createSessionHandler)throw new TypeError("not a valid backend");var i=t[e];if(void 0!==i){if(i.backend===r)return;throw new Error('backend "'+e+'" is already registered')}t[e]={backend:r,priority:o};for(var a=0;a<n.length;a++)if(t[n[a]].priority<=o)return void n.splice(a,0,e);n.push(e)},i=function(e){return r=void 0,o=void 0,a=function(){var r,o,i,a,u,f,l,s,c,y;return function(e,r){var t,n,o,i,a={label:0,sent:function(){if(1&o[0])throw o[1];return o[1]},trys:[],ops:[]};return i={next:u(0),throw:u(1),return:u(2)},"function"==typeof Symbol&&(i[Symbol.iterator]=function(){return this}),i;function u(i){return function(u){return function(i){if(t)throw new TypeError("Generator is already executing.");for(;a;)try{if(t=1,n&&(o=2&i[0]?n.return:i[0]?n.throw||((o=n.return)&&o.call(n),0):n.next)&&!(o=o.call(n,i[1])).done)return o;switch(n=0,o&&(i=[2&i[0],o.value]),i[0]){case 0:case 1:o=i;break;case 4:return a.label++,{value:i[1],done:!1};case 5:a.label++,n=i[1],i=[0];continue;case 7:i=a.ops.pop(),a.trys.pop();continue;default:if(!((o=(o=a.trys).length>0&&o[o.length-1])||6!==i[0]&&2!==i[0])){a=0;continue}if(3===i[0]&&(!o||i[1]>o[0]&&i[1]<o[3])){a.label=i[1];break}if(6===i[0]&&a.label<o[1]){a.label=o[1],o=i;break}if(o&&a.label<o[2]){a.label=o[2],a.ops.push(i);break}o[2]&&a.ops.pop(),a.trys.pop();continue}i=r.call(e,a)}catch(e){i=[6,e],n=0}finally{t=o=0}if(5&i[0])throw i[1];return{value:i[0]?i[1]:void 0,done:!0}}([i,u])}}}(this,(function(p){switch(p.label){case 0:r=0===e.length?n:e,o=[],p.label=1;case 1:p.trys.push([1,9,10,11]),i=function(e){var r="function"==typeof Symbol&&Symbol.iterator,t=r&&e[r],n=0;if(t)return t.call(e);if(e&&"number"==typeof e.length)return{next:function(){return e&&n>=e.length&&(e=void 0),{value:e&&e[n++],done:!e}}};throw new TypeError(r?"Object is not iterable.":"Symbol.iterator is not defined.")}(r),a=i.next(),p.label=2;case 2:if(a.done)return[3,8];if(u=a.value,!(f=t[u]))return[3,7];if(f.initialized)return[2,f.backend];if(f.initializing)throw new Error('backend "'+u+'" is being initialized; cannot initialize multiple times.');if(f.aborted)return[3,7];p.label=3;case 3:return p.trys.push([3,5,6,7]),f.initializing=!0,[4,f.backend.init()];case 4:return p.sent(),f.initialized=!0,[2,f.backend];case 5:return l=p.sent(),o.push({name:u,err:l}),f.aborted=!0,[3,7];case 6:return f.initializing=!1,[7];case 7:return a=i.next(),[3,2];case 8:return[3,11];case 9:return s=p.sent(),c={error:s},[3,11];case 10:try{a&&!a.done&&(y=i.return)&&y.call(i)}finally{if(c)throw c.error}return[7];case 11:throw new Error("no available backend found. ERR: "+o.map((function(e){return"["+e.name+"] "+e.err})).join(", "))}}))},new((i=void 0)||(i=Promise))((function(e,t){function n(e){try{f(a.next(e))}catch(e){t(e)}}function u(e){try{f(a.throw(e))}catch(e){t(e)}}function f(r){var t;r.done?e(r.value):(t=r.value,t instanceof i?t:new i((function(e){e(t)}))).then(n,u)}f((a=a.apply(r,o||[])).next())}));var r,o,i,a},a=new(function(){function e(){this.wasm={},this.webgl={},this.logLevelInternal="warning"}return Object.defineProperty(e.prototype,"logLevel",{get:function(){return this.logLevelInternal},set:function(e){if(void 0!==e){if("string"!=typeof e||-1===["verbose","info","warning","error","fatal"].indexOf(e))throw new Error("Unsupported logging level: "+e);this.logLevelInternal=e}},enumerable:!1,configurable:!0}),e}()),u="undefined"!=typeof BigInt64Array&&"function"==typeof BigInt64Array.from,f="undefined"!=typeof BigUint64Array&&"function"==typeof BigUint64Array.from,l=new Map([["float32",Float32Array],["uint8",Uint8Array],["int8",Int8Array],["uint16",Uint16Array],["int16",Int16Array],["int32",Int32Array],["bool",Uint8Array],["float64",Float64Array],["uint32",Uint32Array]]),s=new Map([[Float32Array,"float32"],[Uint8Array,"uint8"],[Int8Array,"int8"],[Uint16Array,"uint16"],[Int16Array,"int16"],[Int32Array,"int32"],[Float64Array,"float64"],[Uint32Array,"uint32"]]);u&&(l.set("int64",BigInt64Array),s.set(BigInt64Array,"int64")),f&&(l.set("uint64",BigUint64Array),s.set(BigUint64Array,"uint64"));var c=function(){function e(e,r,t){var n,o,i;if("string"==typeof e)if(n=e,i=t,"string"===e){if(!Array.isArray(r))throw new TypeError("A string tensor's data must be a string array.");o=r}else{var a=l.get(e);if(void 0===a)throw new TypeError("Unsupported tensor type: "+e+".");if(Array.isArray(r))o=a.from(r);else{if(!(r instanceof a))throw new TypeError("A "+n+" tensor's data must be type of "+a);o=r}}else if(i=r,Array.isArray(e)){if(0===e.length)throw new TypeError("Tensor type cannot be inferred from an empty array.");var u=typeof e[0];if("string"===u)n="string",o=e;else{if("boolean"!==u)throw new TypeError("Invalid element type of data array: "+u+".");n="bool",o=Uint8Array.from(e)}}else{var f=s.get(e.constructor);if(void 0===f)throw new TypeError("Unsupported type for tensor data: "+e.constructor+".");n=f,o=e}if(void 0===i)i=[o.length];else if(!Array.isArray(i))throw new TypeError("A tensor's dims must be a number array");var c=function(e){for(var r=1,t=0;t<e.length;t++){var n=e[t];if("number"!=typeof n||!Number.isSafeInteger(n))throw new TypeError("dims["+t+"] must be an integer, got: "+n);if(n<0)throw new RangeError("dims["+t+"] must be a non-negative integer, got: "+n);r*=n}return r}(i);if(c!==o.length)throw new Error("Tensor's size("+c+") does not match data length("+o.length+").");this.dims=i,this.type=n,this.data=o,this.size=c}return e.prototype.reshape=function(r){return new e(this.type,this.data,r)},e}(),y=function(e,r,t,n){return new(t||(t=Promise))((function(o,i){function a(e){try{f(n.next(e))}catch(e){i(e)}}function u(e){try{f(n.throw(e))}catch(e){i(e)}}function f(e){var r;e.done?o(e.value):(r=e.value,r instanceof t?r:new t((function(e){e(r)}))).then(a,u)}f((n=n.apply(e,r||[])).next())}))},p=function(e,r){var t,n,o,i,a={label:0,sent:function(){if(1&o[0])throw o[1];return o[1]},trys:[],ops:[]};return i={next:u(0),throw:u(1),return:u(2)},"function"==typeof Symbol&&(i[Symbol.iterator]=function(){return this}),i;function u(i){return function(u){return function(i){if(t)throw new TypeError("Generator is already executing.");for(;a;)try{if(t=1,n&&(o=2&i[0]?n.return:i[0]?n.throw||((o=n.return)&&o.call(n),0):n.next)&&!(o=o.call(n,i[1])).done)return o;switch(n=0,o&&(i=[2&i[0],o.value]),i[0]){case 0:case 1:o=i;break;case 4:return a.label++,{value:i[1],done:!1};case 5:a.label++,n=i[1],i=[0];continue;case 7:i=a.ops.pop(),a.trys.pop();continue;default:if(!((o=(o=a.trys).length>0&&o[o.length-1])||6!==i[0]&&2!==i[0])){a=0;continue}if(3===i[0]&&(!o||i[1]>o[0]&&i[1]<o[3])){a.label=i[1];break}if(6===i[0]&&a.label<o[1]){a.label=o[1],o=i;break}if(o&&a.label<o[2]){a.label=o[2],a.ops.push(i);break}o[2]&&a.ops.pop(),a.trys.pop();continue}i=r.call(e,a)}catch(e){i=[6,e],n=0}finally{t=o=0}if(5&i[0])throw i[1];return{value:i[0]?i[1]:void 0,done:!0}}([i,u])}}},h=function(e){var r="function"==typeof Symbol&&Symbol.iterator,t=r&&e[r],n=0;if(t)return t.call(e);if(e&&"number"==typeof e.length)return{next:function(){return e&&n>=e.length&&(e=void 0),{value:e&&e[n++],done:!e}}};throw new TypeError(r?"Object is not iterable.":"Symbol.iterator is not defined.")},b=function(){function e(e){this.handler=e}return e.prototype.run=function(e,r,t){return y(this,void 0,void 0,(function(){var n,o,i,a,u,f,l,s,y,b,d,w,g,m,v,A,E,T,x,j,S,O,U,k,I,P;return p(this,(function(p){switch(p.label){case 0:if(n={},o={},"object"!=typeof e||null===e||e instanceof c||Array.isArray(e))throw new TypeError("'feeds' must be an object that use input names as keys and OnnxValue as corresponding values.");if(i=!0,"object"==typeof r){if(null===r)throw new TypeError("Unexpected argument[1]: cannot be null.");if(r instanceof c)throw new TypeError("'fetches' cannot be a Tensor");if(Array.isArray(r)){if(0===r.length)throw new TypeError("'fetches' cannot be an empty array.");i=!1;try{for(a=h(r),u=a.next();!u.done;u=a.next()){if("string"!=typeof(v=u.value))throw new TypeError("'fetches' must be a string array or an object.");if(-1===this.outputNames.indexOf(v))throw new RangeError("'fetches' contains invalid output name: "+v+".");n[v]=null}}catch(e){x={error:e}}finally{try{u&&!u.done&&(j=a.return)&&j.call(a)}finally{if(x)throw x.error}}if("object"==typeof t&&null!==t)o=t;else if(void 0!==t)throw new TypeError("'options' must be an object.")}else{f=!1,l=Object.getOwnPropertyNames(r);try{for(s=h(this.outputNames),y=s.next();!y.done;y=s.next())v=y.value,-1!==l.indexOf(v)&&(null===(b=r[v])||b instanceof c)&&(f=!0,i=!1,n[v]=b)}catch(e){S={error:e}}finally{try{y&&!y.done&&(O=s.return)&&O.call(s)}finally{if(S)throw S.error}}if(f){if("object"==typeof t&&null!==t)o=t;else if(void 0!==t)throw new TypeError("'options' must be an object.")}else o=r}}else if(void 0!==r)throw new TypeError("Unexpected argument[1]: must be 'fetches' or 'options'.");try{for(d=h(this.inputNames),w=d.next();!w.done;w=d.next())if(v=w.value,void 0===e[v])throw new Error("input '"+v+"' is missing in 'feeds'.")}catch(e){U={error:e}}finally{try{w&&!w.done&&(k=d.return)&&k.call(d)}finally{if(U)throw U.error}}if(i)try{for(g=h(this.outputNames),m=g.next();!m.done;m=g.next())v=m.value,n[v]=null}catch(e){I={error:e}}finally{try{m&&!m.done&&(P=g.return)&&P.call(g)}finally{if(I)throw I.error}}return[4,this.handler.run(e,n,o)];case 1:for(T in A=p.sent(),E={},A)Object.hasOwnProperty.call(A,T)&&(E[T]=new c(A[T].type,A[T].data,A[T].dims));return[2,E]}}))}))},e.create=function(r,t,n,o){return y(this,void 0,void 0,(function(){var a,u,f,l,s,c,y;return p(this,(function(p){switch(p.label){case 0:if(u={},"string"==typeof r){if(a=r,"object"==typeof t&&null!==t)u=t;else if(void 0!==t)throw new TypeError("'options' must be an object.")}else if(r instanceof Uint8Array){if(a=r,"object"==typeof t&&null!==t)u=t;else if(void 0!==t)throw new TypeError("'options' must be an object.")}else{if(!(r instanceof ArrayBuffer||"undefined"!=typeof SharedArrayBuffer&&r instanceof SharedArrayBuffer))throw new TypeError("Unexpected argument[0]: must be 'path' or 'buffer'.");if(f=r,l=0,s=r.byteLength,"object"==typeof t&&null!==t)u=t;else if("number"==typeof t){if(l=t,!Number.isSafeInteger(l))throw new RangeError("'byteOffset' must be an integer.");if(l<0||l>=f.byteLength)throw new RangeError("'byteOffset' is out of range [0, "+f.byteLength+").");if(s=r.byteLength-l,"number"==typeof n){if(s=n,!Number.isSafeInteger(s))throw new RangeError("'byteLength' must be an integer.");if(s<=0||l+s>f.byteLength)throw new RangeError("'byteLength' is out of range (0, "+(f.byteLength-l)+"].");if("object"==typeof o&&null!==o)u=o;else if(void 0!==o)throw new TypeError("'options' must be an object.")}else if(void 0!==n)throw new TypeError("'byteLength' must be a number.")}else if(void 0!==t)throw new TypeError("'options' must be an object.");a=new Uint8Array(f,l,s)}return c=u.executionProviders||[],y=c.map((function(e){return"string"==typeof e?e:e.name})),[4,i(y)];case 1:return[4,p.sent().createSessionHandler(a,u)];case 2:return[2,new e(p.sent())]}}))}))},e.prototype.startProfiling=function(){this.handler.startProfiling()},e.prototype.endProfiling=function(){this.handler.endProfiling()},Object.defineProperty(e.prototype,"inputNames",{get:function(){return this.handler.inputNames},enumerable:!1,configurable:!0}),Object.defineProperty(e.prototype,"outputNames",{get:function(){return this.handler.outputNames},enumerable:!1,configurable:!0}),e}(),d=exports;for(var w in r)d[w]=r[w];r.__esModule&&Object.defineProperty(d,"__esModule",{value:!0})}();
//# sourceMappingURL=ort-common.node.js.map