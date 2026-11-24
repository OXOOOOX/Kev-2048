/* Browser-local copies of explicitly imported replay files and their catalog rows. */
(function () {
  'use strict';
  const storeName = 'sources';
  const fileFields = ['file', 'runFile', 'resultFile', 'identityFile'];

  function openLibrary() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('kev2048-review-imports', 1);
      request.onupgradeneeded = () => request.result.createObjectStore(storeName, {keyPath: 'key'});
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);
    });
  }

  function transaction(db, mode, action) {
    return new Promise((resolve, reject) => {
      const tx = db.transaction(storeName, mode);
      const request = action(tx.objectStore(storeName));
      tx.oncomplete = () => resolve(request.result);
      tx.onabort = () => reject(tx.error || new Error('浏览器存储写入中断'));
    });
  }

  async function snapshot(file) {
    // Copy the bytes so the browser can still play a game after the original folder moves.
    const blob = new Blob([await file.arrayBuffer()], {type: file.type});
    return {blob, name: file.name, relativePath: file.webkitRelativePath || '', modified: file.lastModified};
  }

  function asFile(stored) {
    return {name: stored.name, webkitRelativePath: stored.relativePath, lastModified: stored.modified,
      size: stored.blob.size, type: stored.blob.type, text: () => stored.blob.text(),
      arrayBuffer: () => stored.blob.arrayBuffer()};
  }

  async function load() {
    if (!window.indexedDB) return [];
    const db = await openLibrary();
    try {
      const records = await transaction(db, 'readonly', store => store.getAll());
      return records.flatMap(({path, files, games}) => {
        const entry = {path};
        for (const [field, stored] of Object.entries(files)) entry[field] = asFile(stored);
        return games.map(game => ({...game, entry}));
      });
    } finally { db.close(); }
  }

  async function save(items) {
    if (!window.indexedDB) throw new Error('此浏览器未启用本地存储');
    const groups = new Map();
    for (const {entry, ...item} of items) {
      if (!groups.has(entry)) groups.set(entry, []);
      groups.get(entry).push(item);
    }
    const db = await openLibrary();
    try {
      for (const [entry, games] of groups) {
        const files = {};
        for (const field of fileFields) {
          if (entry[field]) files[field] = await snapshot(entry[field]);
        }
        const key = entry.file.webkitRelativePath || JSON.stringify([entry.path, games.map(g => g.id).sort()]);
        await transaction(db, 'readwrite', store => store.put({key, path: entry.path, files, games}));
      }
    } finally { db.close(); }
  }

  window.KevImportLibrary = {load, save};
})();
